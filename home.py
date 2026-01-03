import streamlit as st
import pandas as pd
import re
import plotly.express as px

# --- OPPSETT ---
st.set_page_config(page_title="Min utbytte-tracker", layout="wide", page_icon="📈")

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

def smart_fill_name(row):
    """
    Finner selskapsnavnet hvis 'Verdipapir' er tomt.
    Dette skjer ofte på Aksjeutlån og Returprovisjon.
    """
    verdipapir = row.get('Verdipapir', '')
    # Hvis verdipapir finnes og ikke er tomt/NaN, bruk det
    if pd.notna(verdipapir) and str(verdipapir).strip() != "" and str(verdipapir).lower() != "nan":
        return str(verdipapir).strip()
    
    # Hvis tomt, let i transaksjonsteksten
    tekst = str(row.get('Transaksjonstekst', ''))
    
    # 1. Håndter Aksjeutlån (f.eks. "MPC Container Ships - 2024Q4")
    if "aksjeutlån" in str(row.get('Transaksjonstype', '')).lower():
        # Fjern " - 202xQx" på slutten
        clean_text = re.sub(r'\s-\s\d{4}Q\d', '', tekst)
        return clean_text.strip()

    # 2. Håndter Returprovisjon (f.eks. "Returprovisjon for NO... Landkreditt...")
    if "returprovisjon" in tekst.lower():
        # Fjerner "Returprovisjon for [ISIN] "
        clean_text = re.sub(r'Returprovisjon for (NO\d+\s)?', '', tekst)
        return clean_text.strip()
    
    # 3. Håndter generell tekst hvis alt annet feiler
    if tekst.strip() != "":
        return tekst.strip()

    return "Ukjent selskap"

def load_robust_csv(uploaded_file):
    """Robust innlesing av CSV-filer."""
    separators = ['\t', ';', ',']
    encodings = ['utf-16', 'utf-8', 'latin-1', 'iso-8859-1']
    
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
    """Vasker transaksjonslisten og fikser manglende navn."""
    df.columns = df.columns.str.strip()
    
    # Fiks dato
    if 'Bokføringsdag' in df.columns:
        df['Dato'] = pd.to_datetime(df['Bokføringsdag'], errors='coerce')
        df['Måned'] = df['Dato'].dt.strftime('%Y-%m')
        df['År'] = df['Dato'].dt.year
        
    # Rens beløp
    if 'Beløp' in df.columns:
        df['Beløp_Clean'] = df['Beløp'].apply(clean_currency)
    
    # FIKS: Bruk smart navn-utfylling
    df['Verdipapir'] = df.apply(smart_fill_name, axis=1)
        
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

    div_types = ['UTBYTTE', 'Utbetaling aksjeutlån', 'TILBAKEBET. FOND AVG']
    
    # Håndter Reinvest: Inkluder kun hvis positivt (sjelden), eller ignorer de negative
    reinvest = df[
        (df['Transaksjonstype'] == 'REINVESTERT UTBYTTE') & 
        (df['Beløp_Clean'] > 0)
    ].copy()
    
    roc_types = ['TILBAKEBETALING', 'TILBAKEBETALING AV KAPITAL']
    tax_types = ['KUPONGSKATT', 'KORR UTL KUPSKATT']
    
    # Filtrer datasettene
    df_divs = df[df['Transaksjonstype'].isin(div_types)].copy()
    if not reinvest.empty:
        df_divs = pd.concat([df_divs, reinvest])

    df_roc = df[df['Transaksjonstype'].isin(roc_types)].copy()
    df_tax = df[df['Transaksjonstype'].isin(tax_types)].copy()
    
    # Merk typene
    df_divs['Type'] = 'Utbytte'
    df_divs.loc[df_divs['Transaksjonstype'] == 'TILBAKEBET. FOND AVG', 'Type'] = 'Returprovisjon'
    df_divs.loc[df_divs['Transaksjonstype'] == 'Utbetaling aksjeutlån', 'Type'] = 'Aksjeutlån'
    
    df_roc['Type'] = 'Tilbakebetaling'
    
    df_main = pd.concat([df_divs, df_roc])
    
    if df_main.empty:
        return pd.DataFrame()

    # Koble skatt
    if 'Verifikationsnummer' in df_main.columns:
        df_main['Key'] = df_main['Verifikationsnummer'].fillna('Unknown')
        df_tax['Key'] = df_tax['Verifikationsnummer'].fillna('Unknown')
        
        tax_map = df_tax.groupby('Key')['Beløp_Clean'].sum()
        df_main['Kildeskatt'] = df_main['Key'].map(tax_map).fillna(0.0)
    else:
        df_main['Kildeskatt'] = 0.0

    df_main['Brutto_Beløp'] = df_main['Beløp_Clean']
    df_main['Netto_Mottatt'] = df_main['Brutto_Beløp'] + df_main['Kildeskatt']
    
    # Kun positive beløp (cash in)
    df_main = df_main[df_main['Netto_Mottatt'] > 0]
    
    return df_main

# --- HOVEDAPPLIKASJON ---

st.title("💰 Utbytte-dashboard")

# Sidebar
st.sidebar.header("Innstillinger")
konto_type = st.sidebar.selectbox("Kontotype", ["IKZ", "ASK", "AF-konto"])

if konto_type == "AF-konto":
    st.sidebar.warning("⚠️ **AF-konto:** 'Tilbakebetaling' er skattefritt (senker GAV). Vanlig utbytte skattlegges.")
else:
    st.sidebar.success(f"✅ **{konto_type}:** Både utbytte og tilbakebetaling behandles likt (utsatt skatt).")

# Tabs
tab1, tab2 = st.tabs(["📊 Historikk (transaksjoner)", "📷 Nåsituasjon (portefølje)"])

# --- TAB 1: HISTORIKK ---
with tab1:
    st.header("Historisk kontantstrøm")
    uploaded_trans = st.file_uploader("Last opp transaksjons-CSV", type=["csv", "txt"], key="trans")
    
    if uploaded_trans:
        df_raw = load_robust_csv(uploaded_trans)
        if not df_raw.empty:
            df_clean = process_transactions(df_raw)
            df_result = analyze_dividends(df_clean)
            
            if not df_result.empty:
                years = sorted(df_result['År'].dropna().unique(), reverse=True)
                
                # --- VIS TOTALUTVIKLING ---
                if len(years) > 1:
                    st.subheader("📈 Årlig utvikling")
                    yearly_stats = df_result.groupby(['År', 'Type'])['Netto_Mottatt'].sum().reset_index()
                    
                    fig_trend = px.bar(yearly_stats, x='År', y='Netto_Mottatt', color='Type',
                                       title="Utbetalinger år for år", text_auto='.2s',
                                       color_discrete_map={
                                           'Utbytte': '#00CC96', 
                                           'Tilbakebetaling': '#AB63FA',
                                           'Aksjeutlån': '#FFA15A',
                                           'Returprovisjon': '#19D3F3'
                                       })
                    st.plotly_chart(fig_trend, use_container_width=True)
                    st.divider()

                # --- DETALJER PER ÅR ---
                selected_year = st.selectbox("Velg år for detaljer", years)
                df_year = df_result[df_result['År'] == selected_year]
                
                # Metrics
                total = df_year['Netto_Mottatt'].sum()
                
                stats = df_year.groupby('Type')['Netto_Mottatt'].sum()
                
                cols = st.columns(len(stats) + 1)
                cols[0].metric("Totalt (netto)", f"{total:,.0f} NOK")
                
                for i, (type_name, value) in enumerate(stats.items()):
                    cols[i+1].metric(type_name, f"{value:,.0f} NOK")
                
                # Graf
                monthly = df_year.groupby(['Måned', 'Type'])['Netto_Mottatt'].sum().reset_index()
                fig = px.bar(monthly, x='Måned', y='Netto_Mottatt', color='Type',
                             title=f"Utbetalinger per måned ({selected_year})", text_auto='.2s')
                st.plotly_chart(fig, use_container_width=True)
                
                # Tabell (OPPDATERT MED NY BREDDE-PARAM)
                st.write("Transaksjoner:")
                st.dataframe(
                    df_year[['Dato', 'Verdipapir', 'Type', 'Netto_Mottatt', 'Transaksjonstekst']].sort_values('Dato', ascending=False),
                    width="stretch"
                )
            else:
                st.warning("Fant ingen utbytter eller tilbakebetalinger i filen.")

# --- TAB 2: PORTFOLIO ---
with tab2:
    st.header("Porteføljeoversikt")
    method = st.radio("Metode:", ["Last opp CSV (anbefalt)", "Lim inn tekst (backup)"])
    
    df_port = pd.DataFrame()
    
    if method == "Last opp CSV (anbefalt)":
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
            "Est. Utbytte": st.column_config.NumberColumn(label="Est. utbytte", format="%.2f kr", step=0.1),
        }
        
        # OPPDATERT MED NY BREDDE-PARAM
        edited_df = st.data_editor(
            df_port[cols], 
            column_config=column_config, 
            width="stretch"
        )
        
        if 'Antall' in edited_df.columns and 'Est. Utbytte' in edited_df.columns:
            edited_df['Sum utbytte'] = edited_df['Antall'] * edited_df['Est. Utbytte']
            if 'GAV' in edited_df.columns:
                edited_df['YoC %'] = edited_df.apply(lambda x: (x['Est. Utbytte']/x['GAV']*100) if x['GAV']>0 else 0, axis=1)
                
            total = edited_df['Sum utbytte'].sum()
            st.metric("Estimert årlig inntekt", f"{total:,.0f} NOK")
