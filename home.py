import streamlit as st
import pandas as pd
import re
import plotly.express as px

# --- OPPSETT ---
st.set_page_config(page_title="Min Utbytte-Tracker", layout="wide", page_icon="📈")

# --- HJELPEFUNKSJONER ---

def clean_currency(val):
    """Renser tall fra Nordnet (f.eks '1 234,50' -> 1234.50)."""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        # Fjern hard spaces, vanlige mellomrom og bytt komma med punktum
        val = val.replace('\xa0', '').replace(' ', '').replace(',', '.')
        try:
            return float(val)
        except ValueError:
            return 0.0
    return 0.0

def load_robust_csv(uploaded_file):
    """Prøver ulike kodinger for å unngå UnicodeDecodeError."""
    separators = ['\t', ';', ',']
    encodings = ['utf-16', 'utf-8', 'latin-1', 'iso-8859-1']
    
    for enc in encodings:
        for sep in separators:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=sep, decimal=',', encoding=enc, on_bad_lines='skip')
                if df.shape[1] > 1: # Sjekk at vi faktisk fant kolonner
                    return df
            except Exception:
                continue
    return pd.DataFrame()

def parse_clipboard_text(text):
    """Parser tekst limt inn fra Nordnet 'Oversikt'-siden."""
    # Ser etter mønsteret: Navn -> Valuta (NOK) -> Antall -> GAV
    # Dette matcher formatet du sendte tidligere
    pattern = r"([A-Za-z0-9\s\.\-]+)\nNOK\n([\d\s]+)\n([\d\s,]+)"
    matches = re.findall(pattern, text)
    
    data = []
    for m in matches:
        name = m[0].strip()
        count = float(m[1].replace(" ", ""))
        gav = float(m[2].replace(" ", "").replace(",", "."))
        data.append({"Verdipapir": name, "Antall": count, "GAV": gav})
    
    return pd.DataFrame(data)

def analyze_dividends(df):
    """Kobler transaksjoner til utbytte og skatt."""
    if 'Transaksjonstype' not in df.columns:
        return pd.DataFrame()

    div_types = ['UTBYTTE', 'REINVESTERT UTBYTTE', 'Utbetaling aksjeutlån']
    tax_types = ['KUPONGSKATT', 'KORR UTL KUPSKATT']
    
    df_divs = df[df['Transaksjonstype'].isin(div_types)].copy()
    df_tax = df[df['Transaksjonstype'].isin(tax_types)].copy()
    
    if df_divs.empty:
        return pd.DataFrame()

    # Match skatt via Verifikationsnummer
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

def process_transactions(df):
    """Standardiserer transaksjonslisten."""
    df.columns = df.columns.str.strip()
    if 'Bokføringsdag' in df.columns:
        df['Dato'] = pd.to_datetime(df['Bokføringsdag'], errors='coerce')
        df['Måned'] = df['Dato'].dt.strftime('%Y-%m')
        df['År'] = df['Dato'].dt.year
    if 'Beløp' in df.columns:
        df['Beløp_Clean'] = df['Beløp'].apply(clean_currency)
    return df

# --- HOVEDAPPLIKASJON ---

st.title("💰 Utbytte-Dashboard")

# Sidebar
st.sidebar.header("Innstillinger")
st.sidebar.info("Velg kontotype for riktig kontekst:")
konto_type = st.sidebar.selectbox("Kontotype", ["IKZ", "ASK", "AF-Konto"])

# Tabs
tab1, tab2 = st.tabs(["📊 Historikk (CSV)", "📷 Nåsituasjon (Snapshot)"])

# --- TAB 1: TRANSAKSJONER (HISTORIKK) ---
with tab1:
    st.header("Historiske utbytter (Faktisk)")
    st.markdown("Last opp **Transaksjonslisten** din for å se hva du faktisk har tjent.")
    uploaded_trans = st.file_uploader("Last opp CSV", type=["csv", "txt"], key="trans")
    
    if uploaded_trans:
        df_raw = load_robust_csv(uploaded_trans)
        if not df_raw.empty:
            df_clean = process_transactions(df_raw)
            df_divs = analyze_dividends(df_clean)
            
            if not df_divs.empty:
                # Årsvelger
                years = sorted(df_divs['År'].dropna().unique(), reverse=True)
                selected_year = st.selectbox("Velg år", years)
                df_year = df_divs[df_divs['År'] == selected_year]
                
                # Dashbord
                tot = df_year['Netto_Mottatt'].sum()
                c1, c2 = st.columns(2)
                c1.metric(f"Netto Utbytte {selected_year}", f"{tot:,.0f} NOK")
                c2.metric("Antall utbetalinger", len(df_year))
                
                # Graf
                monthly = df_year.groupby('Måned')['Netto_Mottatt'].sum().reset_index()
                fig = px.bar(monthly, x='Måned', y='Netto_Mottatt', title="Månedlig Utbytte", text_auto='.2s')
                fig.update_traces(marker_color='#00CC96')
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(df_year[['Dato', 'Verdipapir', 'Netto_Mottatt', 'Transaksjonstekst']])
            else:
                st.warning("Fant ingen utbytter i filen.")
        else:
            st.error("Klarte ikke å lese filen. Er det en gyldig CSV?")

# --- TAB 2: PORTFOLIO (NÅSITUASJON) ---
with tab2:
    st.header("Portefølje-oversikt")
    st.markdown("Her legger du inn det du eier **akkurat nå** for å beregne Yield on Cost.")
    
    # Velger mellom Lim inn tekst eller Last opp fil
    input_method = st.radio("Hvordan vil du legge inn data?", ["Lim inn tekst (Ctrl+V)", "Last opp CSV"])
    
    df_port = pd.DataFrame()
    
    if input_method == "Lim inn tekst (Ctrl+V)":
        st.info("Gå til Nordnet -> Portefølje -> Oversikt. Merk tabellen (Ctrl+A), kopier (Ctrl+C) og lim inn under.")
        paste_text = st.text_area("Lim inn her:", height=200, placeholder="Navn\nNOK\nAntall\nGAV...")
        
        if paste_text:
            df_port = parse_clipboard_text(paste_text)
            if df_port.empty:
                st.warning("Klarte ikke å finne data i teksten. Sjekk at du kopierte hele tabellen fra Nordnet.")
    
    elif input_method == "Last opp CSV":
        st.info("Last opp CSV-filen fra 'CSV eksport'-knappen på portefølje-siden.")
        uploaded_port = st.file_uploader("Last opp Portefølje-CSV", type=["csv", "txt"], key="port")
        if uploaded_port:
            df_raw_port = load_robust_csv(uploaded_port)
            if not df_raw_port.empty:
                # Enkel vask av portefølje-CSV
                df_raw_port.columns = df_raw_port.columns.str.strip()
                if 'Antall' in df_raw_port.columns:
                    df_raw_port['Antall'] = df_raw_port['Antall'].apply(clean_currency)
                if 'GAV' in df_raw_port.columns:
                    df_raw_port['GAV'] = df_raw_port['GAV'].apply(clean_currency)
                df_port = df_raw_port

    # --- VISNING AV SNAPSHOT ---
    if not df_port.empty:
        st.success(f"Lastet inn {len(df_port)} posisjoner!")
        
        # Legg til kolonner for å leke med tall
        if 'Est. Utbytte' not in df_port.columns:
            df_port['Est. Utbytte'] = 0.0
            
        st.markdown("### 🔮 Estimat for neste 12 mnd")
        st.write("Fyll inn forventet utbytte per aksje i tabellen under:")
        
        # Vis kun relevante kolonner hvis de finnes
        cols = [c for c in ['Verdipapir', 'Navn', 'Antall', 'GAV', 'Est. Utbytte'] if c in df_port.columns]
        edited_df = st.data_editor(df_port[cols])
        
        # Beregninger i sanntid
        if 'Antall' in edited_df.columns and 'Est. Utbytte' in edited_df.columns:
            edited_df['Sum Utbytte'] = edited_df['Antall'] * edited_df['Est. Utbytte']
            
            if 'GAV' in edited_df.columns:
                # Unngå deling på null
                edited_df['YoC %'] = edited_df.apply(
                    lambda x: (x['Est. Utbytte'] / x['GAV'] * 100) if x['GAV'] > 0 else 0, axis=1
                )
            
            total_est = edited_df['Sum Utbytte'].sum()
            
            st.divider()
            c1, c2 = st.columns(2)
            c1.metric("Estimert Årlig Inntekt", f"{total_est:,.0f} NOK")
            
            # Vis tabell med resultater
            st.write("Dine resultater:")
            vis_cols = [c for c in ['Verdipapir', 'Navn', 'Antall', 'GAV', 'YoC %', 'Sum Utbytte'] if c in edited_df.columns]
            st.dataframe(edited_df[vis_cols].style.format({
                'GAV': '{:.2f}',
                'YoC %': '{:.2f}%',
                'Sum Utbytte': '{:.0f}'
            }))
