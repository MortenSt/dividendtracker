import streamlit as st
import pandas as pd
import re
import plotly.express as px
import numpy as np

# --- OPPSETT ---
st.set_page_config(page_title="Min utbytte-tracker", layout="wide", page_icon="📈")

# --- HJELPEFUNKSJONER ---

def clean_currency(val):
    if pd.isna(val) or val == "": return 0.0
    if isinstance(val, (int, float)): return float(val)
    if isinstance(val, str):
        val = val.replace('\xa0', '').replace(' ', '').replace(',', '.')
        try: return float(val)
        except ValueError: return 0.0
    return 0.0

def normalize_string(text):
    if not isinstance(text, str): return str(text)
    text = text.lower()
    suffixes = [r'\sasa$', r'\sas$', r'\sltd$', r'\scorp$', r'\sab$', r'\splc$', r'\sinc$', r'\sclass a$', r'\sa$']
    for suffix in suffixes:
        text = re.sub(suffix, '', text)
    text = re.sub(r'[^a-z0-9]', '', text)
    return text

def smart_fill_name(row):
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
    if tekst.strip() != "": return tekst.strip()
    return "Ukjent selskap"

def load_robust_csv(uploaded_file):
    separators = ['\t', ';', ',']
    encodings = ['utf-16', 'utf-8', 'latin-1', 'iso-8859-1']
    try:
        sample = uploaded_file.read(1024).decode('utf-8', errors='ignore')
        uploaded_file.seek(0)
        if '\t' in sample: separators = ['\t'] + separators
    except: uploaded_file.seek(0)

    for enc in encodings:
        for sep in separators:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=sep, decimal=',', encoding=enc, on_bad_lines='skip')
                if df.shape[1] > 1: return df
            except: continue
    return pd.DataFrame()

def process_transactions(df):
    df.columns = df.columns.str.strip()
    if 'Bokføringsdag' in df.columns:
        df['Dato'] = pd.to_datetime(df['Bokføringsdag'], errors='coerce')
        df['Måned'] = df['Dato'].dt.strftime('%Y-%m')
        df['År'] = df['Dato'].dt.year
    if 'Beløp' in df.columns: df['Beløp_Clean'] = df['Beløp'].apply(clean_currency)
    if 'Antall' in df.columns: df['Antall'] = df['Antall'].apply(clean_currency)
    df['Verdipapir'] = df.apply(smart_fill_name, axis=1)
    return df

def process_portfolio(df):
    df.columns = df.columns.str.strip()
    if 'Verdipapir' not in df.columns and 'Navn' in df.columns: df['Verdipapir'] = df['Navn']
    for col in ['Antall', 'GAV', 'Siste kurs', 'Markedsverdi', 'Verdi', 'Kostpris']:
        if col in df.columns: df[col] = df[col].apply(clean_currency)
    return df

def parse_clipboard_text(text):
    pattern = r"([A-Za-z0-9\s\.\-]+)\nNOK\n([\d\s\xa0]+)\n([\d\s,\.\xa0]+)"
    matches = re.findall(pattern, text)
    data = []
    for m in matches:
        try:
            name = m[0].strip()
            count = clean_currency(m[1])
            gav = clean_currency(m[2])
            data.append({"Verdipapir": name, "Antall": count, "GAV": gav})
        except: continue
    return pd.DataFrame(data)

def detect_frequency_and_volatility(df_stock):
    """
    Analyserer datoene for å finne frekvens.
    FIX: Ignorerer nå 'Utbetaling aksjeutlån' for frekvensberegningen.
    """
    # Filtrer ut kun EKTE utbytter for datoberegning
    real_div_types = ['UTBYTTE', 'TILBAKEBETALING', 'TILBAKEBETALING AV KAPITAL', 'REINVESTERT UTBYTTE']
    
    # Bruk bare rader som er faktiske utbytter
    df_dates = df_stock[df_stock['Transaksjonstype'].isin(real_div_types)]
    
    # Fallback: Hvis vi KUN har aksjeutlån i historikken (ingen utbytter ennå), bruk alt
    if df_dates.empty:
        df_dates = df_stock

    dates = df_dates['Dato'].dt.date.unique()
    dates.sort()
    
    if len(dates) < 2: return "Ukjent", 0, False
    
    recent_dates = dates[-5:]
    diffs = []
    for i in range(1, len(recent_dates)):
        diffs.append((recent_dates[i] - recent_dates[i-1]).days)
        
    avg_diff = sum(diffs) / len(diffs)
    
    if 20 <= avg_diff <= 45: freq, mult = "Månedlig", 12
    elif 70 <= avg_diff <= 110: freq, mult = "Kvartalsvis", 4
    elif 150 <= avg_diff <= 210: freq, mult = "Halvårlig", 2
    elif 330 <= avg_diff <= 400: freq, mult = "Årlig", 1
    else: freq, mult = "Uregelmessig", 0

    # Volatilitet sjekkes fortsatt på ALT (inkludert utlån kan påvirke snittet, men la oss holde det enkelt)
    dps_series = df_stock.apply(lambda r: r['Beløp_Clean']/r['Antall'] if r['Antall']>0 else 0, axis=1)
    mean_dps = dps_series.mean()
    std_dev = dps_series.std()
    
    is_volatile = False
    if mean_dps > 0 and (std_dev / mean_dps) > 0.2: is_volatile = True
        
    return freq, mult, is_volatile

def auto_match_names(history_names, portfolio_names):
    matches = {}
    port_norm = {normalize_string(p): p for p in portfolio_names}
    for h_name in history_names:
        h_norm = normalize_string(h_name)
        if h_norm in port_norm:
            matches[h_name] = port_norm[h_norm]
            continue
        for p_norm_key, p_real in port_norm.items():
            if len(h_norm) > 3 and (h_norm in p_norm_key or p_norm_key in h_norm):
                 matches[h_name] = p_real
                 break
    return matches

def estimate_dividends_from_history(df_history, df_portfolio, mapping_dict, method="smart"):
    if df_history.empty or df_portfolio.empty: return df_portfolio, 0, [], []

    div_types = ['UTBYTTE', 'Utbetaling aksjeutlån', 'TILBAKEBET. FOND AVG', 'TILBAKEBETALING', 'TILBAKEBETALING AV KAPITAL']
    df_divs = df_history[df_history['Transaksjonstype'].isin(div_types)].copy()
    if df_divs.empty: return df_portfolio, 0, [], []

    # Auto-matching
    unique_hist = df_divs['Verdipapir'].unique()
    unique_port = df_portfolio['Verdipapir'].unique()
    auto_matches = auto_match_names(unique_hist, unique_port)
    
    def apply_mapping(name):
        if name in mapping_dict: return mapping_dict[name]
        if name in auto_matches: return auto_matches[name]
        return name

    df_divs['MappedName'] = df_divs['Verdipapir'].apply(apply_mapping)
    
    max_date = df_divs['Dato'].max()
    cutoff_date = max_date - pd.DateOffset(days=380)
    df_recent = df_divs[df_divs['Dato'] >= cutoff_date].copy()
    
    df_divs['DPS'] = df_divs.apply(lambda r: r['Beløp_Clean']/r['Antall'] if r['Antall']>0 else 0, axis=1)
    df_recent['DPS'] = df_recent.apply(lambda r: r['Beløp_Clean']/r['Antall'] if r['Antall']>0 else 0, axis=1)

    est_map = {}
    freq_info = {}
    
    grouped_all = df_divs.groupby('MappedName')
    grouped_recent = df_recent.groupby('MappedName')['DPS'].sum().to_dict()

    for name, group in grouped_all:
        ttm_val = grouped_recent.get(name, 0.0)
        
        # Endret her: detect_frequency bruker nå kun ekte utbytter
        freq_name, multiplier, is_volatile = detect_frequency_and_volatility(group)
        volatility_tag = " ⚠️" if is_volatile else ""
        
        recent_group = df_recent[df_recent['MappedName'] == name]
        avg_dps = recent_group['DPS'].mean() if not recent_group.empty else 0
        
        last_dps = 0
        if not group.empty:
            # For å finne siste utbytte, bør vi kanskje også ignorere aksjeutlån her?
            # Hvis siste transaksjon var aksjeutlån (lite beløp), blir annualiseringen feil.
            # Vi filtrerer group for å finne siste *reelle* utbytte.
            real_divs = group[group['Transaksjonstype'].isin(['UTBYTTE', 'TILBAKEBETALING', 'TILBAKEBETALING AV KAPITAL'])]
            if not real_divs.empty:
                last_payment = real_divs.sort_values('Dato', ascending=False).iloc[0]
                last_dps = last_payment['DPS']
            else:
                # Fallback hvis kun aksjeutlån finnes
                last_payment = group.sort_values('Dato', ascending=False).iloc[0]
                last_dps = last_payment['DPS']

        if method == "ttm":
            est_map[name] = ttm_val
            freq_info[name] = f"TTM"
        elif method == "avg":
            if multiplier > 0 and avg_dps > 0:
                est_map[name] = avg_dps * multiplier
                freq_info[name] = f"{freq_name} (Snitt){volatility_tag}"
            else:
                est_map[name] = ttm_val
                freq_info[name] = "Uregelmessig (TTM)"
        elif method == "smart":
            if multiplier > 0:
                est_map[name] = last_dps * multiplier
                freq_info[name] = f"{freq_name} (Siste){volatility_tag}"
            else:
                est_map[name] = ttm_val
                freq_info[name] = "Uregelmessig (TTM)"

    matched_names = []
    history_names_mapped = set(df_divs['MappedName'].unique())
    port_set_norm = set([normalize_string(n) for n in portfolio_names])
    
    orphans = []
    for h_name in unique_hist:
        mapped = apply_mapping(h_name)
        if normalize_string(mapped) not in port_set_norm:
            orphans.append(h_name)

    def get_estimate(row):
        p_name = row['Verdipapir']
        p_norm = normalize_string(p_name)
        if p_name in est_map:
            matched_names.append(p_name)
            return est_map[p_name]
        for est_name, val in est_map.items():
            if normalize_string(est_name) == p_norm:
                matched_names.append(p_name)
                return val
        return row.get('Est. Utbytte', 0.0)

    def get_freq_text(row):
        p_name = row['Verdipapir']
        p_norm = normalize_string(p_name)
        if p_name in freq_info: return freq_info[p_name]
        for f_name, txt in freq_info.items():
            if normalize_string(f_name) == p_norm: return txt
        return "-"

    df_portfolio['Est. Utbytte'] = df_portfolio.apply(get_estimate, axis=1)
    df_portfolio['Info'] = df_portfolio.apply(get_freq_text, axis=1)
    
    return df_portfolio, len(set(matched_names)), orphans, list(portfolio_names)

def analyze_dividends(df):
    if 'Transaksjonstype' not in df.columns: return pd.DataFrame()

    div_types = ['UTBYTTE', 'Utbetaling aksjeutlån', 'TILBAKEBET. FOND AVG']
    reinvest = df[(df['Transaksjonstype'] == 'REINVESTERT UTBYTTE') & (df['Beløp_Clean'] > 0)].copy()
    roc_types = ['TILBAKEBETALING', 'TILBAKEBETALING AV KAPITAL']
    tax_types = ['KUPONGSKATT', 'KORR UTL KUPSKATT']
    
    df_divs = df[df['Transaksjonstype'].isin(div_types)].copy()
    if not reinvest.empty: df_divs = pd.concat([df_divs, reinvest])
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
    else: df_main['Kildeskatt'] = 0.0

    df_main['Brutto_Beløp'] = df_main['Beløp_Clean']
    df_main['Netto_Mottatt'] = df_main['Brutto_Beløp'] + df_main['Kildeskatt']
    df_main = df_main[df_main['Netto_Mottatt'] > 0]
    return df_main

# --- HOVEDAPPLIKASJON ---

st.title("💰 Utbytte-dashboard")

if 'history_df' not in st.session_state: st.session_state['history_df'] = pd.DataFrame()
if 'mapping' not in st.session_state: st.session_state['mapping'] = {}
if 'orphans' not in st.session_state: st.session_state['orphans'] = []
if 'port_names' not in st.session_state: st.session_state['port_names'] = []

st.sidebar.header("Innstillinger")
konto_type = st.sidebar.selectbox("Kontotype", ["IKZ", "ASK", "AF-konto"])

if konto_type == "AF-konto": st.sidebar.warning("⚠️ **AF-konto:** 'Tilbakebetaling' er skattefritt.")
else: st.sidebar.success(f"✅ **{konto_type}:** Alt behandles likt.")

tab1, tab2 = st.tabs(["📊 Historikk", "📷 Portefølje"])

# --- TAB 1 ---
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
                    yearly_stats = df_result.groupby(['År', 'Type'])['Netto_Mottatt'].sum().reset_index()
                    fig_trend = px.bar(yearly_stats, x='År', y='Netto_Mottatt', color='Type',
                                       title="Utvikling år for år", text_auto='.2s')
                    st.plotly_chart(fig_trend, width="stretch")

                selected_year = st.selectbox("Velg år", years)
                df_year = df_result[df_result['År'] == selected_year]
                
                stats = df_year.groupby('Type')['Netto_Mottatt'].sum()
                cols = st.columns(len(stats) + 1)
                cols[0].metric("Totalt", f"{df_year['Netto_Mottatt'].sum():,.0f} NOK")
                for i, (k, v) in enumerate(stats.items()): cols[i+1].metric(k, f"{v:,.0f} NOK")
                
                monthly = df_year.groupby(['Måned', 'Type'])['Netto_Mottatt'].sum().reset_index()
                fig = px.bar(monthly, x='Måned', y='Netto_Mottatt', color='Type', title=f"Per måned ({selected_year})", text_auto='.2s')
                st.plotly_chart(fig, width="stretch")
                
                st.dataframe(df_year[['Dato', 'Verdipapir', 'Type', 'Netto_Mottatt', 'Transaksjonstekst']].sort_values('Dato', ascending=False), width="stretch")

# --- TAB 2 ---
with tab2:
    st.header("Portefølje & Estimat")
    method = st.radio("Metode:", ["Last opp CSV", "Lim inn tekst"])
    df_port = pd.DataFrame()
    
    if method == "Last opp CSV":
        uploaded_port = st.file_uploader("Last opp 'aksjelister...csv'", type=["csv", "txt"], key="port")
        if uploaded_port:
            df_raw_port = load_robust_csv(uploaded_port)
            if not df_raw_port.empty: df_port = process_portfolio(df_raw_port)
    else:
        paste_text = st.text_area("Lim inn:", height=150)
        if paste_text: df_port = parse_clipboard_text(paste_text)

    if not df_port.empty:
        if 'Est. Utbytte' not in df_port.columns: df_port['Est. Utbytte'] = 0.0
        if 'Info' not in df_port.columns: df_port['Info'] = "-"
            
        col_opt, col_btn = st.columns([2, 1])
        with col_opt:
            est_method = st.radio("Metode:", ["Smart (Siste annualisert)", "Konservativ (Snitt siste år)", "TTM (Sum 12 mnd)"], horizontal=True)
        
        mapping = {"Smart (Siste annualisert)": "smart", "Konservativ (Snitt siste år)": "avg", "TTM (Sum 12 mnd)": "ttm"}
        
        with col_btn:
            st.write("")
            st.write("")
            if not st.session_state['history_df'].empty:
                if st.button("🤖 Beregn nå"):
                    df_port, count, orphans, port_names = estimate_dividends_from_history(
                        st.session_state['history_df'], 
                        df_port, 
                        st.session_state['mapping'],
                        method=mapping[est_method]
                    )
                    st.session_state['orphans'] = orphans
                    st.session_state['port_names'] = port_names
                    if count > 0: st.success(f"Matchet {count} selskaper!")
                    else: st.warning("Fant få/ingen matcher.")
            else: st.info("Mangler historikk (Fane 1).")

        # --- NAVNEKOBLEREN ---
        if st.session_state['orphans']:
            st.warning(f"⚠️ {len(st.session_state['orphans'])} navn matchet ikke automatisk.")
            with st.expander("🔗 Koble manuelt", expanded=True):
                c1, c2, c3 = st.columns([2, 2, 1])
                with c1: selected_orphan = st.selectbox("Ukjent ticker", st.session_state['orphans'])
                with c2: target_stock = st.selectbox("Tilhører aksje", sorted(st.session_state['port_names']))
                with c3:
                    st.write("")
                    st.write("")
                    if st.button("Lagre kobling"):
                        st.session_state['mapping'][selected_orphan] = target_stock
                        st.session_state['orphans'].remove(selected_orphan)
                        st.rerun()
                if st.session_state['mapping']:
                    st.write("Dine koblinger:")
                    st.json(st.session_state['mapping'])
                    if st.button("Nullstill koblinger"):
                        st.session_state['mapping'] = {}
                        st.rerun()

        if not st.session_state['history_df'].empty and df_port is not None:
             df_port, _, _, _ = estimate_dividends_from_history(
                        st.session_state['history_df'], 
                        df_port, 
                        st.session_state['mapping'],
                        method=mapping[est_method]
                    )

        cols = [c for c in ['Verdipapir', 'Antall', 'GAV', 'Est. Utbytte', 'Info'] if c in df_port.columns]
        column_config = {
            "GAV": st.column_config.NumberColumn(format="%.2f kr"),
            "Est. Utbytte": st.column_config.NumberColumn(format="%.2f kr", step=0.1),
            "Info": st.column_config.TextColumn(disabled=True),
        }
        edited_df = st.data_editor(df_port[cols], column_config=column_config, width="stretch")
        
        if 'Antall' in edited_df.columns and 'Est. Utbytte' in edited_df.columns:
            edited_df['Sum utbytte'] = edited_df['Antall'] * edited_df['Est. Utbytte']
            if 'GAV' in edited_df.columns:
                edited_df['YoC %'] = edited_df.apply(lambda x: (x['Est. Utbytte']/x['GAV']*100) if x['GAV']>0 else 0, axis=1)
            total = edited_df['Sum utbytte'].sum()
            st.metric("Estimert årlig inntekt", f"{total:,.0f} NOK")
            res_cols = [c for c in ['Verdipapir', 'Info', 'YoC %', 'Sum utbytte'] if c in edited_df.columns]
            st.dataframe(edited_df[res_cols].style.format({'YoC %': '{:.2f} %', 'Sum utbytte': '{:.0f} kr'}), width="stretch")
