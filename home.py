import streamlit as st
import pandas as pd
import re
import plotly.express as px

# --- KONFIGURASJON ---
st.set_page_config(page_title="Pro Utbytte-Tracker", layout="wide", page_icon="📈")

# --- FUNKSJONER FOR DATAVASK ---

def clean_currency(val):
    """Renser beløp-strenger (f.eks. '1 234,50' -> 1234.50)"""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        # Fjern norske tusenseparatorer (hard space og vanlig space)
        val = val.replace('\xa0', '').replace(' ', '').replace(',', '.')
        try:
            return float(val)
        except ValueError:
            return 0.0
    return 0.0

def identify_separator(file):
    """Prøver å gjette om filen bruker tabulator eller semikolon"""
    try:
        sample = file.read(1024).decode('utf-8', errors='ignore')
        file.seek(0)
        if '\t' in sample:
            return '\t'
        return ';'
    except:
        return ';'

def process_nordnet_csv(uploaded_file):
    sep = identify_separator(uploaded_file)
    # Les filen med automatisk deteksjon
    df = pd.read_csv(uploaded_file, sep=sep, decimal=',')
    
    # 1. Standardiser kolonnenavn (fjerner unødvendig whitespace)
    df.columns = df.columns.str.strip()
    
    # 2. Rens beløpskolonnen
    if 'Beløp' in df.columns:
        df['Beløp_Clean'] = df['Beløp'].apply(clean_currency)
    
    # 3. Identifiser dato
    if 'Bokføringsdag' in df.columns:
        df['Dato'] = pd.to_datetime(df['Bokføringsdag'], errors='coerce')
        df['Måned'] = df['Dato'].dt.strftime('%Y-%m')
        df['År'] = df['Dato'].dt.year

    return df

def analyze_dividends(df):
    """Avansert logikk for å koble utbytte og kildeskatt"""
    
    # Filtrer på relevante transaksjoner
    div_types = ['UTBYTTE', 'REINVESTERT UTBYTTE', 'Utbetaling aksjeutlån']
    tax_types = ['KUPONGSKATT', 'KORR UTL KUPSKATT']
    
    # Lag egne dataframes
    df_divs = df[df['Transaksjonstype'].isin(div_types)].copy()
    df_tax = df[df['Transaksjonstype'].isin(tax_types)].copy()
    
    if df_divs.empty:
        return pd.DataFrame()

    # --- MATCHING-LOGIKK ---
    # Vi prøver å koble skatt til utbytte via 'Verifikationsnummer' hvis det finnes.
    # Hvis ikke, bruker vi Dato + Verdipapir som nøkkel.
    
    df_divs['Key'] = df_divs['Verifikationsnummer'].fillna('Unknown')
    df_tax['Key'] = df_tax['Verifikationsnummer'].fillna('Unknown')
    
    # Summer skatt per nøkkel
    tax_map = df_tax.groupby('Key')['Beløp_Clean'].sum()
    
    # Legg til skatteinfo i utbyttetabellen
    df_divs['Kildeskatt'] = df_divs['Key'].map(tax_map).fillna(0.0)
    
    # For rader uten verifikasjonsnummer, prøv å matche på Dato + Ticker (Fallback)
    # (Dette er avansert, men sikrer at vi fanger opp "løse" skatterader)
    
    # Beregn Netto og Brutto
    # Merk: Beløp_Clean i Nordnet er allerede Netto for utbytte (som regel), 
    # mens skatt er en negativ post. 
    # MEN: Hvis de står på hver sin linje, er Utbytte-linjen ofte Brutto ELLER Netto avhengig av land.
    # I din fil (File 13) var GOOD utbytte +210 og skatt -31.
    # Det betyr at Utbytte-linjen faktisk er NETTO utbetalt til konto, eller BRUTTO før trekk?
    # La oss anta at 'Beløp' er det som faktisk kom inn på konto (Netto).
    # Da er Brutto = Beløp + abs(Kildeskatt).
    
    # Korreksjon basert på dine data:
    # GOOD Utbytte: 210,83 (Positivt inn på konto)
    # GOOD Skatt: -31,62 (Negativt ut av konto)
    # Sum på konto = 210,83 - 31,62 = 179,21 ?? 
    # VENT: I Nordnet-loggen din står begge beløpene. Hvis begge summeres i Saldo,
    # så er 210,83 BRUTTO (før skatt trekkes på neste linje).
    # La oss sjekke 'Saldo'.
    # Utbytte (210,83) øker saldo. Skatt (-31,62) reduserer saldo.
    # Ergo: Utbytte-linjen er BRUTTO (eller delvis brutto), Skatt trekkes separat.
    # Netto mottatt = Utbytte-linje + Skatt-linje (siden skatt er negativ).
    
    df_divs['Brutto_Beløp'] = df_divs['Beløp_Clean'] # Linjen med 'UTBYTTE' er utgangspunktet
    df_divs['Netto_Beløp'] = df_divs['Brutto_Beløp'] + df_divs['Kildeskatt']
    
    return df_divs

# --- HOVEDAPPLIKASJON ---

st.title("💰 Utbytte-Dashboard Pro")
st.markdown("Last opp CSV fra Nordnet (Transaksjoner) for å analysere kontantstrømmen din.")

uploaded_file = st.file_uploader("Last opp CSV-fil", type=["csv", "txt"])

if uploaded_file:
    with st.spinner('Analyserer transaksjoner...'):
        raw_df = process_nordnet_csv(uploaded_file)
        df_result = analyze_dividends(raw_df)
    
    if not df_result.empty:
        # --- TOP LEVEL METRICS ---
        total_netto = df_result['Netto_Beløp'].sum()
        total_brutto = df_result['Brutto_Beløp'].sum()
        total_skatt = df_result['Kildeskatt'].sum() # Negativt tall
        
        # Finn ut hvor mange unike år filen dekker
        years = sorted(df_result['År'].unique())
        selected_year = st.selectbox("Velg år å vise", years, index=len(years)-1)
        
        # Filtrer på år
        df_year = df_result[df_result['År'] == selected_year]
        
        # Metrikker for valgt år
        col1, col2, col3 = st.columns(3)
        col1.metric("Netto Utbytte", f"{df_year['Netto_Beløp'].sum():,.0f} NOK")
        col2.metric("Betalt Kildeskatt", f"{abs(df_year['Kildeskatt'].sum()):,.0f} NOK", help="Skatt trukket ved kilden (f.eks. USA/Tyskland)")
        col3.metric("Antall Utbetalinger", len(df_year))
        
        # --- VISUALISERING 1: MÅNEDLIG INNTEKT ---
        st.subheader("📅 Månedlig Kontantstrøm")
        monthly_stats = df_year.groupby('Måned')['Netto_Beløp'].sum().reset_index()
        
        fig_bar = px.bar(monthly_stats, x='Måned', y='Netto_Beløp', 
                         title=f"Utbetalinger i {selected_year}",
                         text_auto='.2s', color_discrete_sequence=['#00CC96'])
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # --- VISUALISERING 2: HVEM BETALER MEST? ---
        st.subheader("🏆 Top Utbytteaksjer")
        col_chart, col_table = st.columns([2, 1])
        
        top_stocks = df_year.groupby('Verdipapir')[['Netto_Beløp', 'Kildeskatt']].sum().sort_values('Netto_Beløp', ascending=False)
        top_stocks['Skatteprosent'] = (abs(top_stocks['Kildeskatt']) / (top_stocks['Netto_Beløp'] - top_stocks['Kildeskatt'])) * 100
        
        with col_chart:
            fig_pie = px.pie(top_stocks.reset_index().head(10), values='Netto_Beløp', names='Verdipapir', hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col_table:
            st.dataframe(top_stocks.style.format("{:.0f}"), height=400)

        # --- DETALJERT TABELL ---
        st.subheader("📝 Transaksjonslogg (Sammenslått)")
        display_cols = ['Dato', 'Verdipapir', 'Brutto_Beløp', 'Kildeskatt', 'Netto_Beløp', 'Transaksjonstype']
        st.dataframe(df_year[display_cols].sort_values('Dato', ascending=False), use_container_width=True)
        
    else:
        st.warning("Fant ingen utbyttetransaksjoner i denne filen. Sjekk at du lastet opp riktig CSV.")
