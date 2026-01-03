import streamlit as st
import pandas as pd
import re
import plotly.express as px

# --- OPPSETT ---
st.set_page_config(page_title="Min Utbytte-Tracker", layout="wide", page_icon="📈")

# --- HJELPEFUNKSJONER ---

def clean_currency(val):
    """Renser tall fra Nordnet."""
    if pd.isna(val) or val == "":
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        val = val.replace('\xa0', '').replace(' ', '').replace(',', '.')
        try:
            return float(val)
        except ValueError:
            return 0.0
    return 0.0

def load_robust_csv(uploaded_file):
    """Robust innlesing av CSV-filer."""
    separators = ['\t', ';', ',']
    encodings = ['utf-16', 'utf-8', 'latin-1', 'iso-8859-1']
    
    # Prøv å gjette separator
    try:
        sample = uploaded_file.read(1024).decode('utf-8', errors='ignore')
        uploaded_file.seek(0)
        if '\t' in sample:
            separators = ['\t'] + separators
    except:
        uploaded_file.seek(0)

    for enc in encodings:
        for sep in separators:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=sep, decimal=',', encoding=enc, on_bad_lines='skip')
                if df.shape[1] > 1:
                    return df
            except Exception:
                continue
    return pd.DataFrame()

def process_transactions(df):
    """Vasker transaksjonslisten."""
    df.columns = df.columns.str.strip()
    if 'Bokføringsdag' in df.columns:
        df['Dato'] = pd.to_datetime(df['Bokføringsdag'], errors='coerce')
        df['Måned'] = df['Dato'].dt.strftime('%Y-%m')
        df['År'] = df['Dato'].dt.year
    if 'Beløp' in df.columns:
        df['Beløp_Clean'] = df['Beløp'].apply(clean_currency)
    return df

def process_portfolio(df):
    """Vasker porteføljelisten."""
    df.columns = df.columns.str.strip()
    if 'Verdipapir' not in df.columns and 'Navn' in df.columns:
        df['Verdipapir'] = df['Navn']
    
    for col in ['Antall', 'GAV', 'Siste kurs', 'Markedsverdi', 'Verdi', 'Kostpris']:
        if col in df.columns:
            df[col] = df[col].apply(clean_currency)
    return df

def parse_clipboard_text(text):
    """Backup: Parser tekst limt inn manuelt."""
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
    """Kobler utbytte, tilbakebetaling og skatt."""
    if 'Transaksjonstype' not in df.columns:
        return pd.DataFrame()

    # Definer typer
    div_types = ['UTBYTTE', 'REINVESTERT UTBYTTE', 'Utbetaling aksjeutlån']
    roc_types = ['TILBAKEBETALING', 'TILBAKEBETALING AV KAPITAL'] # Return of Capital
    tax_types = ['KUPONGSKATT', 'KORR UTL KUPSKATT']
    
    # Filtrer
    df_divs = df[df['Transaksjonstype'].isin(div_types)].copy()
    df_roc = df[df['Transaksjonstype'].isin(roc_types)].copy()
    df_tax = df[df['Transaksjonstype'].isin(tax_types)].copy()
    
    # Merk typene for senere visualisering
    df_divs['Type'] = 'Utbytte'
    df_roc['Type'] = 'Tilbakebetaling'
    
    # Slå sammen utbytte og tilbakebetaling i en hovedliste
    df_main = pd.concat([df_divs, df_roc])
    
    if df_main.empty:
        return pd.DataFrame()

    # Koble skatt (Gjelder som regel kun utbytte, men vi sjekker alt)
    if 'Verifikationsnummer' in df_main.columns:
        df_main['Key'] = df_main['Verifikationsnummer'].fillna('Unknown')
        df_tax['Key'] = df_tax['Verifikationsnummer'].fillna('Unknown')
        
        tax_map = df_tax.groupby('Key')['Beløp_Clean'].sum()
        df_main['Kildeskatt'] = df_main['Key'].map(tax_map).fillna(0.0)
    else:
        df_main['Kildeskatt'] = 0.0

    df_main['Brutto_Beløp'] = df_main['Beløp_Clean']
    df_main['Netto_Mottatt'] = df_main['Brutto_Beløp'] + df_main['Kildeskatt']
    
    return df_main

# --- HOVEDAPPLIKASJON ---

st.title("💰 Utbytte-Dashboard")

# Sidebar
st.sidebar.header("Innstillinger")
konto_type = st.sidebar.selectbox("Kontotype", ["IKZ", "ASK", "AF-Konto"])

if konto_type == "AF-Konto":
    st.sidebar.warning("⚠️ **AF-Konto:** 'Tilbakebetaling' er skattefritt, men senker din GAV. Vanlig utbytte skattlegges.")
else:
    st.sidebar.success(f"✅ **{konto_type}:** Både utbytte og tilbakebetaling behandles likt (utsatt skatt).")

# Tabs
tab1, tab2 = st.tabs(["📊 Historikk (Transaksjoner)", "📷 Nåsituasjon (Portefølje)"])

# --- TAB 1: HISTORIKK ---
with tab1:
    st.header("Historisk Kontantstrøm")
    uploaded_trans = st.file_uploader("Last opp Transaksjons-CSV", type=["csv", "txt"], key="trans")
    
    if uploaded_trans:
        df_raw = load_robust_csv(uploaded_trans)
        if not df_raw.empty:
            df_clean = process_transactions(df_raw)
            df_result = analyze_dividends(df_clean)
            
            if not df_result.empty:
                years = sorted(df_result['År'].dropna().unique(), reverse=True)
                
                # --- VIS TOTALUTVIKLING HVIS FLERE ÅR ---
                if len(years) > 1:
                    st.subheader("📈 Årlig Utvikling")
                    # Grupper på både År og Type for å vise fordelingen
                    yearly_stats = df_result.groupby(['År', 'Type'])['Netto_Mottatt'].sum().reset_index()
                    
                    fig_trend = px.bar(yearly_stats, x='År', y='Netto_Mottatt', color='Type',
                                       title="Utbetalinger år for år", text_auto='.2s',
                                       color_discrete_map={'Utbytte': '#00CC96', 'Tilbakebetaling': '#AB63FA'})
                    st.plotly_chart(fig_trend, use_container_width=True)
                    st.divider()

                # --- DETALJER PER ÅR ---
                selected_year = st.selectbox("Velg år for detaljer", years)
                df_year = df_result[df_result['År'] == selected_year]
                
                # Metrics
                tot_utbytte = df_year[df_year['Type'] == 'Utbytte']['Netto_Mottatt'].sum()
                tot_tilbake = df_year[df_year['Type'] == 'Tilbakebetaling']['Netto_Mottatt'].sum()
                tot_total = tot_utbytte + tot_tilbake
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Totalt Mottatt (Cash)", f"{tot_total:,.0f} NOK")
                c2.metric("Herav Utbytte", f"{tot_utbytte:,.0f} NOK")
                c3.metric("Herav Tilbakebetaling", f"{tot_tilbake:,.0f} NOK", 
                          help="På AF-konto er dette skattefritt, men senker GAV.")
                
                # Graf
                monthly = df_year.groupby(['Måned', 'Type'])['Netto_Mottatt'].sum().reset_index()
                fig = px.bar(monthly, x='Måned', y='Netto_Mottatt', color='Type',
                             title=f"Utbetalinger per måned ({selected_year})", text_auto='.2s',
                             color_discrete_map={'Utbytte': '#00CC96', 'Tilbakebetaling': '#AB63FA'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Tabell
                st.write("Transaksjoner:")
                st.dataframe(df_year[['Dato', 'Verdipapir', 'Type', 'Netto_Mottatt', 'Transaksjonstekst']].sort_values('Dato', ascending=False))
            else:
                st.warning("Fant ingen utbytter eller tilbakebetalinger i filen.")

# --- TAB 2: PORTFOLIO ---
with tab2:
    st.header("Portefølje-oversikt")
    method = st.radio("Metode:", ["Last opp CSV (Anbefalt)", "Lim inn tekst (Backup)"])
    
    df_port = pd.DataFrame()
    
    if method == "Last opp CSV (Anbefalt)":
        uploaded_port = st.file_uploader("Last opp 'aksjelister...csv'", type=["csv", "txt"], key="port")
        if uploaded_port:
            df_raw_port = load_robust_csv(uploaded_port)
            if not df_raw_port.empty:
                df_port = process_portfolio(df_raw_port)
    else:
        paste_text = st.text_area("Lim inn fra Nordnet:", height=150)
        if paste_text:
            df_port = parse_clipboard_text(paste_text)

    if not df_port.empty:
        if 'Est. Utbytte' not in df_port.columns:
            df_port['Est. Utbytte'] = 0.0
            
        st.markdown("### 🔮 Estimat")
        
        cols = [c for c in ['Verdipapir', 'Antall', 'GAV', 'Est. Utbytte'] if c in df_port.columns]
        
        column_config = {
            "GAV": st.column_config.NumberColumn(format="%.2f kr"),
            "Est. Utbytte": st.column_config.NumberColumn(format="%.2f kr", step=0.1),
        }
        
        edited_df = st.data_editor(df_port[cols], column_config=column_config, use_container_width=True)
        
        if 'Antall' in edited_df.columns and 'Est. Utbytte' in edited_df.columns:
            edited_df['Sum Utbytte'] = edited_df['Antall'] * edited_df['Est. Utbytte']
            if 'GAV' in edited_df.columns:
                edited_df['YoC %'] = edited_df.apply(lambda x: (x['Est. Utbytte']/x['GAV']*100) if x['GAV']>0 else 0, axis=1)
                
            total = edited_df['Sum Utbytte'].sum()
            st.metric("Estimert Årlig Inntekt", f"{total:,.0f} NOK")
