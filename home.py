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
    
    # Prøv å finne Markedsverdi eller beregne den
    # Fond bruker ofte 'Kostpris' istedenfor GAV
    if 'GAV' not in df.columns and 'Kostpris' in df.columns:
        df['GAV'] = df['Kostpris']

    # Fond bruker ofte 'Verdi NOK' istedenfor Markedsverdi
    if 'Markedsverdi' not in df.columns:
        if 'Verdi' in df.columns: df['Markedsverdi'] = df['Verdi']
        elif 'Verdi NOK' in df.columns: df['Markedsverdi'] = df['Verdi NOK']

    for col in ['Antall', 'GAV', 'Siste kurs', 'Markedsverdi', 'Verdi', 'Kostpris', 'Verdi NOK']:
        if col in df.columns: df[col] = df[col].apply(clean_currency)
    
    # Fallback
    if 'Markedsverdi' not in df.columns:
        if 'Antall' in df.columns and 'Siste kurs' in df.columns:
            df['Markedsverdi'] = df['Antall'] * df['Siste kurs']
        else:
            df['Markedsverdi'] = 0.0 
            
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
            data.append({"Verdipapir": name, "Antall": count, "GAV": gav, "Markedsverdi": 0.0})
        except: continue
    return pd.DataFrame(data)

def detect_frequency_and_volatility(df_stock):
    real_div_types = ['UTBYTTE', 'TILBAKEBETALING', 'TILBAKEBETALING AV KAPITAL', 'REINVESTERT UTBYTTE']
    df_dates = df_stock[df_stock['Transaksjonstype'].isin(real_div_types)]
    if df_dates.empty: df_dates = df_stock
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
    if df_history.empty or df_portfolio.empty: return df_portfolio, 0, [], [], []
    div_types = ['UTBYTTE', 'Utbetaling aksjeutlån', 'TILBAKEBET. FOND AVG', 'TILBAKEBETALING', 'TILBAKEBETALING AV KAPITAL']
    df_divs = df_history[df_history['Transaksjonstype'].isin(div_types)].copy()
    if df_divs.empty: return df_portfolio, 0, [], [], []
    unique_hist = sorted(df_divs['Verdipapir'].unique())
    unique_port = sorted(df_portfolio['Verdipapir'].unique()) if not df_portfolio.empty else []
    auto_matches = auto_match_names(unique_hist, unique_port)
    def apply_mapping(name): return mapping_dict.get(name, name)
    df_divs['MappedName'] = df_divs['Verdipapir'].apply(apply_mapping)
    df_divs['FinalName'] = df_divs['MappedName'].apply(lambda x: auto_matches.get(x, x))
    max_date = df_divs['Dato'].max()
    cutoff_date = max_date - pd.DateOffset(days=380)
    df_recent = df_divs[df_divs['Dato'] >= cutoff_date].copy()
    df_divs['DPS'] = df_divs.apply(lambda r: r['Beløp_Clean']/r['Antall'] if r['Antall']>0 else 0, axis=1)
    df_recent['DPS'] = df_recent.apply(lambda r: r['Beløp_Clean']/r['Antall'] if r['Antall']>0 else 0, axis=1)
    est_map = {}
    freq_info = {}
    grouped_all = df_divs.groupby('FinalName')
    grouped_recent = df_recent.groupby('FinalName')['DPS'].sum().to_dict()
    for name, group in grouped_all:
        ttm_val = grouped_recent.get(name, 0.0)
        freq_name, multiplier, is_volatile = detect_frequency_and_volatility(group)
        volatility_tag = " ⚠️" if is_volatile else ""
        recent_group = df_recent[df_recent['FinalName'] == name]
        avg_dps = recent_group['DPS'].mean() if not recent_group.empty else 0
        last_dps = 0
        if not group.empty:
            real_divs = group[group['Transaksjonstype'].isin(['UTBYTTE', 'TILBAKEBETALING', 'TILBAKEBETALING AV KAPITAL'])]
            last_payment = real_divs.sort_values('Dato', ascending=False).iloc[0] if not real_divs.empty else group.sort_values('Dato', ascending=False).iloc[0]
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
    port_set_norm = set([normalize_string(n) for n in unique_port])
    orphans = []
    for h_name in unique_hist:
        final_mapped = apply_mapping(h_name)
        final_mapped = auto_matches.get(final_mapped, final_mapped)
        if normalize_string(final_mapped) not in port_set_norm:
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
    return df_portfolio, len(set(matched_names)), orphans, unique_port, unique_hist

def analyze_dividends(df, mapping_dict):
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
    df_main['Verdipapir'] = df_main['Verdipapir'].apply(lambda x: mapping_dict.get(x, x))
    return df_main

def analyze_capital_gains(df_hist, mapping_dict, manual_adjustments=None):
    if df_hist.empty: return pd.DataFrame(), [], pd.DataFrame()
    trade_types = [
        'KJØP', 'KJØPT', 'SALG', 'SOLGT', 'KÖP', 'SÄLJ', 
        'TEGNING', 'EMISJON', 'TILDELING', 'OVERFØRSEL', 'INNLØSNING',
        'SPLITT INNLEGG VP', 'SPLITT UTTAK VP'
    ]
    df_trades = df_hist[df_hist['Transaksjonstype'].str.upper().isin(trade_types)].copy()
    if manual_adjustments:
        adj_rows = []
        for adj in manual_adjustments:
            adj_rows.append({'Dato': pd.Timestamp.min, 'Verdipapir': adj['name'], 'Transaksjonstype': 'KJØP (Manuell Start)', 'Beløp_Clean': -abs(adj['cost']), 'Antall': adj['qty']})
        if adj_rows:
            df_adj = pd.DataFrame(adj_rows)
            df_trades = pd.concat([df_adj, df_trades], ignore_index=True)
    if df_trades.empty: return pd.DataFrame(), [], pd.DataFrame()
    df_trades['Verdipapir'] = df_trades['Verdipapir'].apply(lambda x: mapping_dict.get(x, x))
    sales = df_trades[df_trades['Transaksjonstype'].str.upper().isin(['SALG', 'SÄLJ', 'SOLGT'])]
    has_sales = sales['Verdipapir'].unique()
    gains = df_trades.groupby('Verdipapir')['Beløp_Clean'].sum().reset_index()
    gains.columns = ['Verdipapir', 'Handelsresultat']
    return gains, has_sales, df_trades

# --- HOVEDAPPLIKASJON ---

st.title("💰 Utbytte-dashboard")

if 'history_df' not in st.session_state: st.session_state['history_df'] = pd.DataFrame()
if 'portfolio_df' not in st.session_state: st.session_state['portfolio_df'] = pd.DataFrame()
if 'mapping' not in st.session_state: st.session_state['mapping'] = {}
if 'orphans' not in st.session_state: st.session_state['orphans'] = []
if 'port_names' not in st.session_state: st.session_state['port_names'] = []
if 'hist_names' not in st.session_state: st.session_state['hist_names'] = []
if 'manual_adj' not in st.session_state: st.session_state['manual_adj'] = [] 

st.sidebar.header("Innstillinger")
konto_type = st.sidebar.selectbox("Kontotype", ["IKZ", "ASK", "AF-konto"])
if konto_type == "AF-konto": st.sidebar.warning("⚠️ **AF-konto:** 'Tilbakebetaling' er skattefritt.")
else: st.sidebar.success(f"✅ **{konto_type}:** Alt behandles likt.")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Historikk", "📷 Portefølje", "🏆 Toppliste", "🧩 Analyse"])

# --- TAB 1 ---
with tab1:
    st.header("Historisk kontantstrøm")
    uploaded_trans_files = st.file_uploader("Last opp transaksjons-CSV (kan velge flere)", type=["csv", "txt"], key="trans", accept_multiple_files=True)
    
    if uploaded_trans_files:
        df_list = []
        for f in uploaded_trans_files:
            df_temp = load_robust_csv(f)
            if not df_temp.empty:
                df_clean = process_transactions(df_temp)
                df_list.append(df_clean)
        
        if df_list:
            df_total = pd.concat(df_list, ignore_index=True).drop_duplicates()
            st.session_state['history_df'] = df_total
            
            df_result = analyze_dividends(df_total, st.session_state['mapping'])
            if not df_result.empty:
                years = sorted(df_result['År'].dropna().unique(), reverse=True)
                if len(years) > 1:
                    yearly_stats = df_result.groupby(['År', 'Type'])['Netto_Mottatt'].sum().reset_index()
                    fig_trend = px.bar(yearly_stats, x='År', y='Netto_Mottatt', color='Type', title="Utvikling år for år", text_auto='.2s')
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
    method = st.radio("Metode:", ["Last opp CSV (Aksjer/Fond)", "Lim inn tekst"])
    df_port = pd.DataFrame()
    
    if method == "Last opp CSV (Aksjer/Fond)":
        uploaded_port_files = st.file_uploader("Last opp Aksje- og Fondsfiler", type=["csv", "txt"], key="port", accept_multiple_files=True)
        if uploaded_port_files:
            df_p_list = []
            for f in uploaded_port_files:
                df_temp = load_robust_csv(f)
                if not df_temp.empty:
                    df_proc = process_portfolio(df_temp)
                    df_p_list.append(df_proc)
            
            if df_p_list:
                df_port = pd.concat(df_p_list, ignore_index=True).drop_duplicates()
                
    else:
        paste_text = st.text_area("Lim inn:", height=150)
        if paste_text: df_port = parse_clipboard_text(paste_text)

    if not df_port.empty:
        st.session_state['portfolio_df'] = df_port.copy()

        if 'Est. Utbytte' not in df_port.columns: df_port['Est. Utbytte'] = 0.0
        if 'Info' not in df_port.columns: df_port['Info'] = "-"
        col_opt, col_btn = st.columns([2, 1])
        with col_opt:
            est_method = st.radio("Metode:", ["Smart (Siste annualisert)", "Konservativ (Snitt siste år)", "TTM (Sum 12 mnd)"], horizontal=True)
        mapping = {"Smart (Siste annualisert)": "smart", "Konservativ (Snitt siste år)": "avg", "TTM (Sum 12 mnd)": "ttm"}
        
        if not st.session_state['history_df'].empty and df_port is not None:
             df_port, count, orphans, port_names, hist_names = estimate_dividends_from_history(
                        st.session_state['history_df'], 
                        df_port, 
                        st.session_state['mapping'],
                        method=mapping[est_method]
                    )
             st.session_state['orphans'] = orphans
             st.session_state['port_names'] = port_names
             st.session_state['hist_names'] = hist_names
             st.session_state['portfolio_df'] = df_port.copy()

        with col_btn:
            st.write("")
            st.write("")
            if not st.session_state['history_df'].empty: st.button("🤖 Oppdater beregning")

        with st.expander("🔗 Koble navn / Overstyr automatikk", expanded=bool(st.session_state['orphans'])):
            c1, c2, c3 = st.columns([2, 2, 1])
            with c1:
                all_hist = sorted(st.session_state['hist_names'])
                default_ix = 0
                if st.session_state['orphans'] and st.session_state['orphans'][0] in all_hist:
                    default_ix = all_hist.index(st.session_state['orphans'][0])
                selected_ticker = st.selectbox("Ticker / Navn fra historikk", all_hist, index=default_ix)
                current_map = st.session_state['mapping'].get(selected_ticker, None)
                if current_map: st.caption(f"Manuelt koblet til: **{current_map}**")
                else:
                    matches = auto_match_names([selected_ticker], st.session_state['port_names'])
                    if selected_ticker in matches: st.caption(f"🤖 Auto: **{matches[selected_ticker]}**")
                    else: st.caption("⚠️ Ikke koblet")
            with c2:
                suggestions = [""] + sorted(st.session_state['port_names'])
                target_port = st.selectbox("Koble mot portefølje...", suggestions)
                custom_text = st.text_input("...eller skriv nytt navn:", value=target_port if target_port else "")
            with c3:
                st.write("")
                st.write("")
                if st.button("Lagre kobling"):
                    if custom_text.strip():
                        st.session_state['mapping'][selected_ticker] = custom_text.strip()
                        st.rerun()
            if st.session_state['mapping']:
                st.write("---")
                if st.button("Slett alle koblinger"):
                    st.session_state['mapping'] = {}
                    st.rerun()

        cols = [c for c in ['Verdipapir', 'Antall', 'GAV', 'Est. Utbytte', 'Info', 'Markedsverdi'] if c in df_port.columns]
        column_config = {
            "GAV": st.column_config.NumberColumn(format="%.2f kr"),
            "Est. Utbytte": st.column_config.NumberColumn(format="%.2f kr", step=0.1),
            "Markedsverdi": st.column_config.NumberColumn(format="%.0f kr"),
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

# --- TAB 3: TOPPLISTE ---
with tab3:
    st.header("🏆 Toppliste: Totalt")
    if not st.session_state['history_df'].empty:
        df_hist = st.session_state['history_df'].copy()
        df_divs = analyze_dividends(df_hist, st.session_state['mapping'])
        df_gains, has_sales_list, df_raw_trades = analyze_capital_gains(df_hist, st.session_state['mapping'], st.session_state['manual_adj'])
        
        if not df_divs.empty:
            df_divs['NormKey'] = df_divs['Verdipapir'].apply(normalize_string)
            key_to_display = {}
            for key, group in df_divs.groupby('NormKey'):
                key_to_display[key] = max(group['Verdipapir'].unique(), key=len)
            df_divs['DisplayName'] = df_divs['NormKey'].map(key_to_display)
            
            all_display_names = sorted(df_divs['DisplayName'].unique())
            with st.expander("🛠️ Ser du duplikater? Slå dem sammen her"):
                c1, c2, c3 = st.columns([2, 2, 1])
                with c1: src_name = st.selectbox("Navn som skal flettes:", all_display_names)
                with c2: 
                    targets = [n for n in all_display_names if n != src_name]
                    target_name = st.selectbox("...inn i dette navnet:", targets)
                with c3:
                    st.write("")
                    st.write("")
                    if st.button("Slå sammen"):
                        st.session_state['mapping'][src_name] = target_name
                        st.rerun()
            
            with st.expander("🛠️ Juster inngangsverdi / Glemte kjøp"):
                c1, c2, c3 = st.columns([2, 1, 1])
                with c1: adj_name = st.selectbox("Velg aksje:", all_display_names)
                with c2: adj_cost = st.number_input("Hva betalte du totalt? (kr)", min_value=0.0, step=1000.0)
                with c3:
                    st.write("")
                    st.write("")
                    if st.button("Legg til kostnad"):
                        st.session_state['manual_adj'].append({'name': adj_name, 'cost': adj_cost, 'qty': 0})
                        st.success("Lagt til!")
                        st.rerun()
                if st.session_state['manual_adj']:
                    st.write("Dine justeringer:")
                    st.write(st.session_state['manual_adj'])
                    if st.button("Nullstill justeringer"):
                        st.session_state['manual_adj'] = []
                        st.rerun()

            years = sorted(df_divs['År'].dropna().unique(), reverse=True)
            filter_year = st.selectbox("Velg periode", ["Alle år"] + list(years))
            if filter_year != "Alle år": df_view = df_divs[df_divs['År'] == filter_year]
            else: df_view = df_divs
            
            total_divs = df_view.groupby('DisplayName')['Netto_Mottatt'].sum().reset_index()
            total_divs.columns = ['Selskap', 'Utbytte']
            merged = total_divs.copy()
            
            df_port_curr = st.session_state['portfolio_df']
            port_value_map = {}
            if not df_port_curr.empty and 'Markedsverdi' in df_port_curr.columns:
                for idx, row in df_port_curr.iterrows():
                    p_name = row['Verdipapir']
                    val = row['Markedsverdi']
                    norm = normalize_string(p_name)
                    if norm in key_to_display:
                        disp = key_to_display[norm]
                        port_value_map[disp] = port_value_map.get(disp, 0) + val
            merged['Markedsverdi'] = merged['Selskap'].map(port_value_map).fillna(0)

            if not df_gains.empty and filter_year == "Alle år":
                df_gains['NormKey'] = df_gains['Verdipapir'].apply(normalize_string)
                df_gains['DisplayName'] = df_gains['NormKey'].map(key_to_display).fillna(df_gains['Verdipapir'])
                total_gains = df_gains.groupby('DisplayName')['Handelsresultat'].sum().reset_index()
                merged = pd.merge(merged, total_gains, left_on='Selskap', right_on='DisplayName', how='outer')
                merged['Selskap'] = merged['Selskap'].fillna(merged['DisplayName'])
                merged = merged.drop(columns=['DisplayName'])
            else: merged['Handelsresultat'] = 0.0

            merged['Utbytte'] = merged['Utbytte'].fillna(0)
            merged['Handelsresultat'] = merged['Handelsresultat'].fillna(0)
            merged['Markedsverdi'] = merged['Markedsverdi'].fillna(0)

            current_holdings = []
            if 'port_names' in st.session_state:
                current_holdings = [normalize_string(x) for x in st.session_state['port_names']]
            has_sales_norm = [normalize_string(x) for x in has_sales_list]

            def get_status(name):
                n_norm = normalize_string(name)
                in_portfolio = n_norm in current_holdings
                has_sold = n_norm in has_sales_norm
                if in_portfolio and has_sold: return "🟡 Delvis solgt"
                if in_portfolio: return "🟢 Eies"
                return "🔴 Avsluttet"
            merged['Status'] = merged['Selskap'].apply(get_status)
            
            def calc_total(row): return row['Utbytte'] + row['Handelsresultat'] + row['Markedsverdi']
            if filter_year == "Alle år": merged['Totalavkastning'] = merged.apply(calc_total, axis=1)
            else: merged['Totalavkastning'] = np.nan

            merged = merged.sort_values('Utbytte', ascending=False).reset_index(drop=True)
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader(f"Utbytte-kongene ({filter_year})")
                fig = px.bar(merged.head(20), x='Utbytte', y='Selskap', color='Status', orientation='h', title="Topp 20 Utbytte", text_auto='.2s', color_discrete_map={"🟢 Eies": "#00CC96", "🟡 Delvis solgt": "#FFA15A", "🔴 Avsluttet": "#EF553B"})
                fig.update_layout(yaxis={'categoryorder':'total ascending'}) 
                st.plotly_chart(fig, width="stretch")
            with col2:
                st.subheader("Fasit (Total)")
                cols_to_show = ['Selskap', 'Status', 'Utbytte']
                if filter_year == "Alle år": cols_to_show += ['Totalavkastning']
                st.dataframe(merged[cols_to_show].style.format({'Utbytte': '{:,.0f} kr', 'Totalavkastning': '{:,.0f} kr'}, na_rep="-"), width="stretch")
                if filter_year == "Alle år": st.caption("* Totalavkastning = Utbytte + (Salg - Kjøp) + Dagens Verdi.")

            st.divider()
            st.subheader("🔍 Dykk ned i tallene")
            selected_xray = st.selectbox("Velg selskap for å se detaljer:", merged['Selskap'].unique())
            if selected_xray:
                target_norm_key = normalize_string(selected_xray)
                df_divs_xray = df_divs[df_divs['NormKey'] == target_norm_key].copy()
                df_trades_xray = pd.DataFrame()
                if not df_raw_trades.empty:
                    df_raw_trades['NormKey'] = df_raw_trades['Verdipapir'].apply(normalize_string)
                    df_trades_xray = df_raw_trades[df_raw_trades['NormKey'] == target_norm_key].copy()
                c_a, c_b = st.columns(2)
                with c_a:
                    st.write("**Utbytter:**")
                    st.dataframe(df_divs_xray[['Dato', 'Verdipapir', 'Beløp_Clean', 'Transaksjonstekst']].sort_values('Dato', ascending=False), width="stretch")
                with c_b:
                    st.write("**Handel & Beholdning:**")
                    if not df_trades_xray.empty:
                        df_trades_xray = df_trades_xray.sort_values('Dato')
                        def get_flow(row):
                            typ = str(row['Transaksjonstype']).upper()
                            amt = row['Antall']
                            val = row['Beløp_Clean']
                            if 'INNLEGG' in typ: return abs(amt)
                            if 'UTTAK' in typ: return -abs(amt)
                            if val < 0: return abs(amt) 
                            if val > 0: return -abs(amt) 
                            return 0 
                        df_trades_xray['Flow'] = df_trades_xray.apply(get_flow, axis=1)
                        df_trades_xray['Beholdning'] = df_trades_xray['Flow'].cumsum()
                        fig_holding = px.line(df_trades_xray, x='Dato', y='Beholdning', title="Aksjebeholdning over tid (Est.)", markers=True)
                        st.plotly_chart(fig_holding, width="stretch")
                        st.dataframe(df_trades_xray[['Dato', 'Verdipapir', 'Transaksjonstype', 'Beløp_Clean', 'Antall']].sort_values('Dato', ascending=False), width="stretch")
                        tot_net = df_trades_xray['Beløp_Clean'].sum()
                        curr_val = port_value_map.get(selected_xray, 0)
                        st.metric("Handelsresultat (Cashflow)", f"{tot_net:,.0f} kr")
                        st.metric("Totalavkastning (inkl utbytte & verdi)", f"{tot_net + curr_val + df_divs_xray['Netto_Mottatt'].sum():,.0f} kr")
                        min_hold = df_trades_xray['Beholdning'].min()
                        if min_hold < 0: st.error(f"⚠️ Grafen går under null ({min_hold} aksjer).")
                    else: st.info("Ingen handler funnet.")
        else: st.warning("Fant ingen utbytter.")
    else: st.info("Mangler historikk.")

# --- TAB 4: ANALYSE ---
with tab4:
    st.header("🧩 Portefølje-analyse")
    
    if not st.session_state['portfolio_df'].empty:
        df_an = st.session_state['portfolio_df'].copy()
        
        # Beregn grunnlag
        df_an['Total Kost'] = df_an['Antall'] * df_an['GAV']
        df_an['Total Inntekt'] = df_an['Antall'] * df_an['Est. Utbytte']
        
        # Filtrer vekk null-verdier for renere grafer
        df_an_cost = df_an[df_an['Total Kost'] > 0]
        df_an_val = df_an[df_an['Markedsverdi'] > 0]
        df_an_inc = df_an[df_an['Total Inntekt'] > 0]
        
        # --- RAD 1: Verdi vs Kost ---
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("💰 Hvor er pengene nå? (Markedsverdi)")
            fig_val = px.pie(df_an_val, values='Markedsverdi', names='Verdipapir', hole=0.4)
            st.plotly_chart(fig_val, width="stretch")
            
        with c2:
            st.subheader("💸 Hvor satset du? (Kostpris)")
            fig_cost = px.pie(df_an_cost, values='Total Kost', names='Verdipapir', hole=0.4)
            st.plotly_chart(fig_cost, width="stretch")
            
        st.divider()
        
        # --- RAD 2: Inntektskilden ---
        st.subheader("💵 Hvem betaler lønna di? (Utbytte-fordeling)")
        if not df_an_inc.empty:
            c_inc1, c_inc2 = st.columns([2, 1])
            with c_inc1:
                fig_inc = px.pie(df_an_inc, values='Total Inntekt', names='Verdipapir', hole=0.4)
                fig_inc.update_traces(textinfo='percent+label')
                st.plotly_chart(fig_inc, width="stretch")
            with c_inc2:
                st.write("Topp 5 bidragsytere:")
                top_inc = df_an_inc.sort_values('Total Inntekt', ascending=False).head(5)
                st.dataframe(top_inc[['Verdipapir', 'Total Inntekt']].style.format({'Total Inntekt': '{:,.0f} kr'}), hide_index=True)
        else:
            st.info("Ingen estimerte utbytter funnet enda.")

        st.divider()

        # --- RAD 3: Yield on Cost (Bonus) ---
        st.subheader("📈 Yield on Cost (Din rente) vs. Dagens rente")
        
        # Sjekk at vi ikke deler på 0
        df_an = df_an[df_an['GAV'] > 0].copy()
        
        if 'Siste kurs' in df_an.columns:
             df_an = df_an[df_an['Siste kurs'] > 0].copy()
             df_an['YoC'] = (df_an['Est. Utbytte'] / df_an['GAV']) * 100
             df_an['Yield'] = (df_an['Est. Utbytte'] / df_an['Siste kurs']) * 100
             
             # Vasker data
             df_an = df_an.replace([np.inf, -np.inf], 0)
             df_yield = df_an[df_an['Total Inntekt'] > 0].sort_values('YoC', ascending=False)
             
             if not df_yield.empty:
                df_melt = df_yield.melt(id_vars=['Verdipapir'], value_vars=['YoC', 'Yield'], var_name='Type', value_name='Prosent')
                fig_yoc = px.bar(df_melt, x='Verdipapir', y='Prosent', color='Type', barmode='group',
                                 title="Avkastning på investert kapital (YoC) vs Markedsrente",
                                 color_discrete_map={"YoC": "#00CC96", "Yield": "#636EFA"})
                st.plotly_chart(fig_yoc, width="stretch")
        
    else:
        st.info("Last opp portefølje i 'Portefølje'-fanen for å se analyse.")
