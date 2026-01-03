import streamlit as st
import pandas as pd
import re
import plotly.express as px
import numpy as np

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
    """Finner selskapsnavnet hvis 'Verdipapir' er tomt."""
    verdipapir = row.get('Verdipapir', '')
    if pd.notna(verdipapir) and str(verdipapir).strip() != "" and str(verdipapir).lower() != "nan":
        return str(verdipapir).strip()
    
    tekst = str(row.get('Transaksjonstekst', ''))
    
    if "aksjeutlån" in str(row.get('Transaksjonstype', '')).lower():
        clean_text = re.sub(r'\s-\s\d{4}Q\d', '', tekst)
        return clean_text.strip()

    if "returprovisjon" in tekst.lower():
        clean_text = re.sub(r'Returprovisjon for (NO\d+\s)?', '', tekst)
        return clean_text.strip()
    
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
    """Vasker transaksjonslisten."""
    df.columns = df.columns.str.strip()
    if 'Bokføringsdag' in df.columns:
        df['Dato'] = pd.to_datetime(df['Bokføringsdag'], errors='coerce')
        df['Måned'] = df['Dato'].dt.strftime('%Y-%m')
        df['År'] = df['Dato'].dt.year
    if 'Beløp' in df.columns:
        df['Beløp_Clean'] = df['Beløp'].apply(clean_currency)
    if 'Antall' in df.columns:
        df['Antall'] = df['Antall'].apply(clean_currency)
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

def detect_frequency_and_volatility(df_stock):
    """
    Analyserer datoene for å finne frekvens, og beløpene for å finne volatilitet.
    Returnerer: (frekvens_navn, multiplier, is_volatile)
    """
    dates = df_stock['Dato'].dt.date.unique()
    dates.sort()
    
    if len(dates) < 2:
        return "Ukjent", 0, False
    
    # 1. Frekvensanalyse (siste 5)
    recent_dates = dates[-5:]
    diffs = []
    for i in range(1, len(recent_dates)):
        diffs.append((recent_dates[i] - recent_dates[i-1]).days)
        
    avg_diff = sum(diffs) / len(diffs)
    
    if 20 <= avg_diff <= 40:
        freq, mult = "Månedlig", 12
    elif 75 <= avg_diff <= 105:
        freq, mult = "Kvartalsvis", 4
    elif 150 <= avg_diff <= 210:
        freq, mult = "Halvårlig", 2
    elif 330 <= avg_diff <= 400:
        freq, mult = "Årlig", 1
    else:
        freq, mult = "Uregelmessig", 0

    # 2. Volatilitetsanalyse (Sjekk om utbyttet hopper opp og ned)
    # Beregn DPS for hver utbetaling for å sammenligne
    dps_series = df_stock.apply(lambda r: r['Beløp_Clean']/r['Antall'] if r['Antall']>0 else 0, axis=1)
    
    # Hvis standardavviket er høyt i forhold til snittet, er det volatilt (ekstraordinært?)
    mean_dps = dps_series.mean()
    std_dev = dps_series.std()
    
    is_volatile = False
    if mean_dps > 0 and (std_dev / mean_dps) > 0.2: # Hvis variasjonen er mer enn 20%
        is_volatile = True
        
    return freq, mult, is_volatile

def estimate_dividends_from_history(df_history, df_portfolio, method="smart"):
    """
    Beregner estimert utbytte.
    method="ttm": Sum siste 12 mnd.
    method="smart": Annualisert siste utbytte (aggressiv).
    method="avg": Annualisert gjennomsnitt siste 12 mnd (konservativ, bra for shipping).
    """
    if df_history.empty or df_portfolio.empty:
        return df_portfolio

    div_types = ['UTBYTTE', 'Utbetaling aksjeutlån', 'TILBAKEBET. FOND AVG', 'TILBAKEBETALING', 'TILBAKEBETALING AV KAPITAL']
    df_divs = df_history[df_history['Transaksjonstype'].isin(div_types)].copy()
    
    if df_divs.empty:
        return df_portfolio

    # Siste 12 mnd
    max_date = df_divs['Dato'].max()
    cutoff_date = max_date - pd.DateOffset(days=365)
    df_recent = df_divs[df_divs['Dato'] >= cutoff_date].copy()
    
    df_divs['DPS'] = df_divs.apply(lambda r: r['Beløp_Clean']/r['Antall'] if r['Antall']>0 else 0, axis=1)
    df_recent['DPS'] = df_recent.apply(lambda r: r['Beløp_Clean']/r['Antall'] if r['Antall']>0 else 0, axis=1)

    est_map = {}
    freq_info = {}

    grouped_all = df_divs.groupby('Verdipapir')
    grouped_recent = df_recent.groupby('Verdipapir')['DPS'].sum().to_dict()

    for name, group in grouped_all:
        ttm_val = grouped_recent.get(name, 0.0)
        
        # Kjør analyse
        freq_name, multiplier, is_volatile = detect_frequency_and_volatility(group)
        volatility_tag = " ⚠️ Variabelt" if is_volatile else ""
        
        # Beregn snitt-DPS siste 12 mnd (for 'avg' metode)
        recent_group = df_recent[df_recent['Verdipapir'] == name]
        avg_dps = recent_group['DPS'].mean() if not recent_group.empty else 0
        
        # Finn siste DPS
        last_dps = 0
        if not group.empty:
            last_payment = group.sort_values('Dato', ascending=False).iloc[0]
            last_dps = last_payment['DPS']

        # VELG METODE
        if method == "ttm":
            est_map[name] = ttm_val
            freq_info[name] = f"TTM (Sum 12 mnd)"
            
        elif method == "avg":
            # Konservativ: Snitt av siste års betalinger * frekvens
            if multiplier > 0 and avg_dps > 0:
                est_val = avg_dps * multiplier
                est_map[name] = est_val
                freq_info[name] = f"{freq_name} (Snitt){volatility_tag}"
            else:
                est_map[name] = ttm_val # Fallback
                freq_info[name] = "Uregelmessig (TTM)"

        elif method == "smart":
            # Aggressiv: Siste * frekvens
            if multiplier > 0:
                est_val = last_dps * multiplier
                est_map[name] = est_val
                freq_info[name] = f"{freq_name} (Siste){volatility_tag}"
            else:
                est_map[name] = ttm_val
                freq_info[name] = "Uregelmessig (TTM)"

    def get_estimate(row):
        return est_map.get(row['Verdipapir'], row.get('Est. Utbytte', 0.0))

    def get_freq_text(row):
        return freq_info.get(row['Verdipapir'], "-")

    df_portfolio['Est. Utbytte'] = df_portfolio.apply(get_estimate, axis=1)
    df_portfolio['Info'] = df_portfolio.apply(get_freq_text, axis=1)
    
    return df_portfolio, len(est_map)

def analyze_dividends(df):
    """Kobler utbytte, tilbakebetaling og skatt."""
    if 'Transaksjonstype' not in df.columns:
        return pd.DataFrame()

    div_types = ['UTBYTTE', 'Utbetaling aksjeutlån', 'TILBAKEBET. FOND AVG']
    reinvest = df[(df['Transaksjonstype'] == 'REINVESTERT UTBYTTE') & (df['Beløp_Clean'] > 0)].copy()
    roc_types = ['TILBAKEBETALING', 'TILBAKEBETALING AV KAPITAL']
    tax_types = ['KUPONGSKATT', 'KORR UTL KUPSKATT']
    
    df_divs = df[df['Transaksjonstype'].isin(div_types)].copy()
    if not reinvest.empty:
        df_divs = pd.concat([df_divs, reinvest])
    df_roc = df[df['Transaksjonstype'].isin(roc_types)].copy()
    df_tax = df[df['Transaksjonstype'].isin(tax_types)].copy()
    
    df_divs['Type'] = 'Utbytte'
    df_divs.loc[df_divs['Transaksjonstype'] == 'TILBAKEBET. FOND AVG', 'Type'] = 'Returprovisjon'
    df_divs.loc[df_divs['Transaksjonstype'] == 'Utbetaling aksjeutlån', 'Type'] = 'Aksjeutlån'
    df_roc['Type'] = 'Tilbakebetaling'
    
    df_main = pd.concat([df_divs, df_roc])
    if df_main.empty: return pd.DataFrame()

    if 'Verifikationsnummer' in df_main.columns:
        df_main['Key'] = df_main['Verifikationsnummer'].fillna('Unknown')
        df_tax['Key'] = df_tax['Verifikationsnummer'].fillna('Unknown')
        tax_map = df_tax.groupby('Key')['Beløp_Clean'].sum()
        df_main['Kildeskatt'] = df_main['Key'].map(tax_map).fillna(0.0)
    else:
        df_main['Kildeskatt'] = 0.0

    df_main['Brutto_Beløp'] = df_main['Beløp_Clean']
    df_main['Netto_Mottatt'] = df_main['Brutto_Beløp'] + df_main['Kildeskatt']
    df_main = df_main[df_main['Netto_Mottatt'] > 0]
    
    return df_main

# --- HOVEDAPPLIKASJON ---

st.title("💰 Utbytte-dashboard")

if 'history_df' not in st.session_state:
    st.session_state['history_df'] = pd.DataFrame()

st.sidebar.header("Innstillinger")
konto_type = st.sidebar.selectbox("Kontotype", ["IKZ", "ASK", "AF-konto"])

if konto_type == "AF-konto":
    st.sidebar.warning("⚠️ **AF-konto:** 'Tilbakebetaling' er skattefritt (senker GAV).")
else:
    st.sidebar.success(f"✅ **{konto_type}:** Alt behandles likt (utsatt skatt).")

tab1, tab2 = st.tabs(["📊 Historikk (transaksjoner)", "📷 Nåsituasjon (portefølje)"])

# --- TAB 1: HISTORIKK ---
with tab1:
    st.header("Historisk kontantstrøm")
    uploaded_trans = st.file_uploader("Last opp transaksjons-CSV", type=["csv", "txt"], key="trans")
    
    if uploaded_trans:
        df_raw = load_robust_csv(uploaded_trans)
        if not df_raw.empty:
            df_clean = process_transactions(df_raw)
            st.session_state['history_df'] = df_clean
            
            df_result = analyze_dividends(df_clean)
            
            if not df_result.empty:
                years = sorted(df_result['År'].dropna().unique(), reverse=True)
                
                if len(years) > 1:
                    st.subheader("📈 Årlig utvikling")
                    yearly_stats = df_result.groupby(['År', 'Type'])['Netto_Mottatt'].sum().reset_index()
                    fig_trend = px.bar(yearly_stats, x='År', y='Netto_Mottatt', color='Type',
                                       title="Utbetalinger år for år", text_auto='.2s',
                                       color_discrete_map={'Utbytte': '#00CC96', 'Tilbakebetaling': '#AB63FA', 'Aksjeutlån': '#FFA15A', 'Returprovisjon': '#19D3F3'})
                    st.plotly_chart(fig_trend, use_container_width=True)
                    st.divider()

                selected_year = st.selectbox("Velg år for detaljer", years)
                df_year = df_result[df_result['År'] == selected_year]
                
                total = df_year['Netto_Mottatt'].sum()
                stats = df_year.groupby('Type')['Netto_Mottatt'].sum()
                
                cols = st.columns(len(stats) + 1)
                cols[0].metric("Totalt (netto)", f"{total:,.0f} NOK")
                for i, (type_name, value) in enumerate(stats.items()):
                    cols[i+1].metric(type_name, f"{value:,.0f} NOK")
                
                monthly = df_year.groupby(['Måned', 'Type'])['Netto_Mottatt'].sum().reset_index()
                fig = px.bar(monthly, x='Måned', y='Netto_Mottatt', color='Type',
                             title=f"Utbetalinger per måned ({selected_year})", text_auto='.2s')
                st.plotly_chart(fig, use_container_width=True)
                
                st.write("Transaksjoner:")
                st.dataframe(df_year[['Dato', 'Verdipapir', 'Type', 'Netto_Mottatt', 'Transaksjonstekst']].sort_values('Dato', ascending=False), width="stretch")
            else:
                st.warning("Fant ingen utbytter.")

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
        if 'Est. Utbytte' not in df_port.columns: df_port['Est. Utbytte'] = 0.0
        if 'Info' not in df_port.columns: df_port['Info'] = "-"
            
        st.markdown("### 🔮 Estimat")
        
        col_opt, col_btn = st.columns([2, 1])
        with col_opt:
            est_method = st.radio("Velg beregningsmetode:", 
                                  ["Smart (Siste annualisert)", "Konservativ (Snitt siste år)", "TTM (Sum 12 mnd)"],
                                  help="Smart ganger opp siste utbytte (risikabelt i shipping). Konservativ bruker snittet av alle utbetalinger siste år.",
                                  horizontal=True)
        
        mapping = {"Smart (Siste annualisert)": "smart", "Konservativ (Snitt siste år)": "avg", "TTM (Sum 12 mnd)": "ttm"}
        
        with col_btn:
            st.write("") 
            st.write("") 
            if not st.session_state['history_df'].empty:
                if st.button("🤖 Beregn estimat"):
                    df_port, count = estimate_dividends_from_history(st.session_state['history_df'], df_port, method=mapping[est_method])
                    if count > 0: st.success(f"Oppdaterte {count} selskaper!")
                    else: st.warning("Ingen match.")
            else:
                st.info("Last opp transaksjoner først.")

        cols = [c for c in ['Verdipapir', 'Antall', 'GAV', 'Est. Utbytte', 'Info'] if c in df_port.columns]
        column_config = {
            "GAV": st.column_config.NumberColumn(format="%.2f kr"),
            "Est. Utbytte": st.column_config.NumberColumn(label="Est. utbytte (per aksje)", format="%.2f kr", step=0.1),
            "Info": st.column_config.TextColumn(label="Metode & Volatilitet", disabled=True),
        }
        
        edited_df = st.data_editor(df_port[cols], column_config=column_config, width="stretch")
        
        if 'Antall' in edited_df.columns and 'Est. Utbytte' in edited_df.columns:
            edited_df['Sum utbytte'] = edited_df['Antall'] * edited_df['Est. Utbytte']
            if 'GAV' in edited_df.columns:
                edited_df['YoC %'] = edited_df.apply(lambda x: (x['Est. Utbytte']/x['GAV']*100) if x['GAV']>0 else 0, axis=1)
                
            total = edited_df['Sum utbytte'].sum()
            c1, c2 = st.columns(2)
            c1.metric("Estimert årlig inntekt", f"{total:,.0f} NOK")
            
            st.write("Resultater per aksje:")
            res_cols = [c for c in ['Verdipapir', 'Info', 'YoC %', 'Sum utbytte'] if c in edited_df.columns]
            st.dataframe(edited_df[res_cols].style.format({'YoC %': '{:.2f} %', 'Sum utbytte': '{:.0f} kr'}), width="stretch")
