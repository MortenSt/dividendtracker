import streamlit as st
import pandas as pd
import re
import plotly.express as px

# --- OPPSETT ---
st.set_page_config(page_title="Min Utbytte-Tracker", layout="wide", page_icon="📈")

# --- HJELPEFUNKSJONER ---

def clean_currency(val):
    """
    Renser tall fra Nordnet.
    Håndterer:
    - '1 234,50' (vanlig mellomrom)
    - '1\xa0234,50' (hard space fra HTML/Excel)
    - '71,2315' (komma desimal)
    """
    if pd.isna(val) or val == "":
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        # Fjern alle typer mellomrom og bytt komma med punktum
        val = val.replace('\xa0', '').replace(' ', '').replace(',', '.')
        try:
            return float(val)
        except ValueError:
            return 0.0
    return 0.0

def load_robust_csv(uploaded_file):
    """
    Prøver å lese CSV-filen uansett om den er lagret med
    semikolon, tabulator eller komma, og uansett tegnsett.
    """
    separators = ['\t', ';', ',']
    encodings = ['utf-16', 'utf-8', 'latin-1', 'iso-8859-1']
    
    # Prøv å gjette separator basert på første linje
    try:
        sample = uploaded_file.read(1024).decode('utf-8', errors='ignore')
        uploaded_file.seek(0)
        if '\t' in sample:
            separators = ['\t'] + separators # Prioriter tab hvis funnet
    except:
        uploaded_file.seek(0)

    for enc in encodings:
        for sep in separators:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=sep, decimal=',', encoding=enc, on_bad_lines='skip')
                # Sjekk at vi faktisk fikk kolonner (mer enn 1)
                if df.shape[1] > 1:
                    return df
            except Exception:
                continue
                
    return pd.DataFrame()

def process_transactions(df):
    """Vasker transaksjonslisten (Historikk)."""
    df.columns = df.columns.str.strip()
    
    # Standardiser dato
    if 'Bokføringsdag' in df.columns:
        df['Dato'] = pd.to_datetime(df['Bokføringsdag'], errors='coerce')
        df['Måned'] = df['Dato'].dt.strftime('%Y-%m')
        df['År'] = df['Dato'].dt.year
        
    # Rens beløp
    if 'Beløp' in df.columns:
        df['Beløp_Clean'] = df['Beløp'].apply(clean_currency)
        
    return df

def process_portfolio(df):
    """Vasker porteføljelisten (Nåsituasjon)."""
    df.columns = df.columns.str.strip()
    
    # Map 'Navn' til 'Verdipapir' hvis det er det filen bruker
    if 'Verdipapir' not in df.columns and 'Navn' in df.columns:
        df['Verdipapir'] = df['Navn']
        
    # Kolonner som må være tall
    numeric_cols = ['Antall', 'GAV', 'Siste kurs', 'Markedsverdi', 'Verdi', 'Kostpris']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_currency)
            
    return df

def parse_clipboard_text(text):
    """Backup: Parser tekst limt inn manuelt."""
    # Regex for: Navn -> Valuta -> Antall -> GAV
    pattern = r"([A-Za-z0-9\s\.\-]+)\nNOK\n([\d\s\xa0]+)\n([\d\s,\.\xa0]+)"
    matches = re.findall(pattern, text)
    
    data = []
    for m in matches:
        try:
            name = m[0].strip()
            count = clean_currency(m[1])
            gav = clean_currency(m[2])
            data.append({"Verdipapir": name, "Antall": count, "GAV": gav})
        except:
            continue
    return pd.DataFrame(data)

def analyze_dividends(df):
    """Kobler utbytte og skatt."""
    if 'Transaksjonstype' not in df.columns:
        return pd.DataFrame()

    div_types = ['UTBYTTE', 'REINVESTERT UTBYTTE', 'Utbetaling aksjeutlån']
    tax_types = ['KUPONGSKATT', 'KORR UTL KUPSKATT']
    
    df_divs = df[df['Transaksjonstype'].isin(div_types)].copy()
    df_tax = df[df['Transaksjonstype'].isin(tax_types)].copy()
    
    if df_divs.empty:
        return pd.DataFrame()

    # Koble skatt via Verifikationsnummer
    if 'Verifikationsnummer' in df_divs.columns:
        df_divs['Key'] = df_divs['Verifikationsnummer'].fillna('Unknown')
        df_tax['Key'] = df_tax['Verifikationsnummer'].fillna('Unknown')
        tax_map = df_tax.groupby('Key')['Beløp_Clean'].sum()
        df_divs['Kildeskatt'] = df_divs['Key'].map(tax_map).fillna(0.0)
    else:
        df_divs['Kildeskatt'] = 0.0

    df_divs['Brutto_Utbytte'] = df_divs['Beløp_Clean']
    df_divs['Netto_Mottatt'] = df_divs['Brutto_Utbytte'] + df_divs['Kildeskatt']
    
    return df_divs

# --- HOVEDAPPLIKASJON ---

st.title("💰 Utbytte-Dashboard")

# Sidebar
st.sidebar.header("Innstillinger")
konto_type = st.sidebar.selectbox("Kontotype", ["IKZ", "ASK", "AF-Konto"])
st.sidebar.info(f"Viser logikk for: **{konto_type}**")

# Tabs
tab1, tab2 = st.tabs(["📊 Historikk (Transaksjoner)", "📷 Nåsituasjon (Portefølje)"])

# --- TAB 1: HISTORIKK ---
with tab1:
    st.header("Historisk Utbytte")
    st.markdown("For å se hva du **faktisk** har fått utbetalt:")
    st.info("1. Gå til Nordnet -> Transaksjoner og notaer -> Eksporter til CSV.\n2. Last opp filen `transactions-and-notes-export.csv` her.")
    
    uploaded_trans = st.file_uploader("Last opp Transaksjons-CSV", type=["csv", "txt"], key="trans")
    
    if uploaded_trans:
        df_raw = load_robust_csv(uploaded_trans)
        if not df_raw.empty:
            df_clean = process_transactions(df_raw)
            df_divs = analyze_dividends(df_clean)
            
            if not df_divs.empty:
                years = sorted(df_divs['År'].dropna().unique(), reverse=True)
                selected_year = st.selectbox("Velg år", years)
                df_year = df_divs[df_divs['År'] == selected_year]
                
                # Metrics
                tot = df_year['Netto_Mottatt'].sum()
                c1, c2 = st.columns(2)
                c1.metric(f"Netto Utbytte {selected_year}", f"{tot:,.0f} NOK")
                c2.metric("Antall Utbetalinger", len(df_year))
                
                # Graf
                monthly = df_year.groupby('Måned')['Netto_Mottatt'].sum().reset_index()
                fig = px.bar(monthly, x='Måned', y='Netto_Mottatt', title="Utbetalinger per måned", text_auto='.2s')
                fig.update_traces(marker_color='#00CC96')
                st.plotly_chart(fig, use_container_width=True)
                
                # Tabell
                st.dataframe(df_year[['Dato', 'Verdipapir', 'Netto_Mottatt', 'Transaksjonstekst']].sort_values('Dato', ascending=False))
            else:
                st.warning("Fant ingen utbytter i denne filen.")
        else:
            st.error("Klarte ikke å lese filen. Sjekk at det er en gyldig CSV.")

# --- TAB 2: PORTFOLIO ---
with tab2:
    st.header("Portefølje-oversikt")
    st.markdown("Her legger du inn det du eier **akkurat nå** for å beregne Yield on Cost.")
    
    # Velg metode - CSV er nå standard
    method = st.radio("Metode:", ["Last opp CSV (Anbefalt)", "Lim inn tekst (Backup)"])
    
    df_port = pd.DataFrame()
    
    if method == "Last opp CSV (Anbefalt)":
        st.info("💡 **Tips:** Gå til Nordnet -> Portefølje -> 'CSV eksport'-knappen (til høyre over tabellen).")
        uploaded_port = st.file_uploader("Last opp filen `aksjelister_konto...csv`", type=["csv", "txt"], key="port")
        
        if uploaded_port:
            df_raw_port = load_robust_csv(uploaded_port)
            if not df_raw_port.empty:
                df_port = process_portfolio(df_raw_port)
                st.success(f"Leste inn {len(df_port)} aksjer fra filen!")
    
    else: # Backup metode
        st.write("Gå til Nordnet -> Portefølje -> Merk alt (Ctrl+A) -> Kopier (Ctrl+C).")
        paste_text = st.text_area("Lim inn her:", height=150)
        if paste_text:
            df_port = parse_clipboard_text(paste_text)

    # --- VISNING OG ESTIMATER ---
    if not df_port.empty:
        # Legg til kolonner for estimat
        if 'Est. Utbytte' not in df_port.columns:
            df_port['Est. Utbytte'] = 0.0
            
        st.markdown("### 🔮 Estimat for neste 12 mnd")
        st.caption("Fyll inn forventet utbytte per aksje i kolonnen 'Est. Utbytte' under for å se totalen.")
        
        # Velg hvilke kolonner vi vil vise/redigere
        cols = [c for c in ['Verdipapir', 'Antall', 'GAV', 'Siste kurs', 'Est. Utbytte'] if c in df_port.columns]
        
        # Konfigurer kolonner for data_editor
        column_config = {
            "GAV": st.column_config.NumberColumn(format="%.2f kr"),
            "Est. Utbytte": st.column_config.NumberColumn(format="%.2f kr", step=0.1),
            "Antall": st.column_config.NumberColumn(format="%.0f")
        }
        
        edited_df = st.data_editor(df_port[cols], column_config=column_config, use_container_width=True)
        
        # Beregninger
        if 'Antall' in edited_df.columns and 'Est. Utbytte' in edited_df.columns:
            edited_df['Sum Utbytte'] = edited_df['Antall'] * edited_df['Est. Utbytte']
            
            if 'GAV' in edited_df.columns:
                edited_df['YoC %'] = edited_df.apply(
                    lambda x: (x['Est. Utbytte'] / x['GAV'] * 100) if x['GAV'] > 0 else 0, axis=1
                )
                
            total_est = edited_df['Sum Utbytte'].sum()
            
            st.divider()
            c1, c2 = st.columns(2)
            c1.metric("Estimert Årlig Inntekt", f"{total_est:,.0f} NOK")
            
            # Vis resultat-tabell
            st.write("Resultater per aksje:")
            res_cols = [c for c in ['Verdipapir', 'YoC %', 'Sum Utbytte'] if c in edited_df.columns]
            
            st.dataframe(edited_df[res_cols].style.format({
                'YoC %': '{:.2f} %',
                'Sum Utbytte': '{:.0f} kr'
            }), use_container_width=True)
