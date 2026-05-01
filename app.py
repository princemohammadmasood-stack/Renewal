"""
B2C License Renewal Prediction App
===================================
Streamlit app that scores new company data for renewal likelihood.
Trains on historical data, accepts Excel uploads, outputs renewal scores.

Setup:
  pip install streamlit pandas numpy scikit-learn openpyxl xlsxwriter
  Place "B2C_Renewal_Decision_Data.xlsx" in the same directory as this script.
  Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
REF_DATE = pd.Timestamp('2026-01-01')
TRAINING_DATA = "B2C_Renewal_Decision_Data.xlsx"

RISK_MAP = {'Low': 1, 'Medium': 2, 'High': 3, 'Override': 4}
APPROVAL_MAP = {'No': 0, 'Yes': 1}

FEATURE_COLS = [
    'Visa Allocation', 'Total No of Activities', 'No of Groups Opted',
    'Number of Shareholders', 'Average Shareholder Age', 'Renewal Year',
    'Latest Year Transactions', 'Cumulative Transactions', 'Transaction Trend',
    'Weighted Activity Risk Encoded', 'Weighted Nationality Risk Encoded',
    'Third-party Approval Encoded',
    'Zero Transactions Flag', 'Log Latest Transactions',
    'Log Cumulative Transactions', 'Engagement Ratio',
    'Transactions per Visa', 'Combined Risk Score'
]

# Geographical risk ratings (embedded)
GEO_RISK = {
    'Afghanistan': 'High', 'Aland Islands': 'Low', 'Albania': 'Medium', 'Algeria': 'High',
    'American Samoa': 'Low', 'Andorra': 'Medium', 'Angola': 'High', 'Anguilla': 'Low',
    'Antarctica': 'Low', 'Antigua and Barbuda': 'Medium', 'Argentina': 'High', 'Armenia': 'Low',
    'Aruba': 'Low', 'Australia': 'Low', 'Austria': 'Low', 'Azerbaijan': 'Medium',
    'Bahamas': 'Low', 'Bahrain': 'Medium', 'Bangladesh': 'Medium', 'Barbados': 'Medium',
    'Belarus': 'High', 'Belgium': 'Low', 'Belize': 'Medium', 'Benin': 'Medium',
    'Bermuda': 'Low', 'Bhutan': 'Medium', 'Bolivia': 'High', 'Bosnia and Herzegovina': 'High',
    'Botswana': 'Low', 'Brazil': 'Medium', 'British Virgin Islands': 'High', 'Brunei': 'Low',
    'Bulgaria': 'High', 'Burkina Faso': 'Medium', 'Burundi': 'High', 'Cambodia': 'High',
    'Cameroon': 'High', 'Canada': 'Low', 'Cape Verde': 'Medium', 'Cayman Islands': 'Medium',
    'Central African Republic': 'High', 'Chad': 'Medium', 'Chile': 'Low', 'China': 'Medium',
    'Colombia': 'Medium', 'Comoros': 'Medium', 'Cook Islands': 'Medium', 'Costa Rica': 'Low',
    'Croatia': 'Medium', 'Cuba': 'High', 'Cyprus': 'Low', 'Czech Republic': 'Low',
    'Democratic Republic of the Congo': 'High', 'Denmark': 'Low', 'Djibouti': 'Medium',
    'Dominica': 'Low', 'Dominican Republic': 'Medium', 'East Timor': 'Low', 'Ecuador': 'Medium',
    'Egypt': 'Medium', 'El Salvador': 'Medium', 'Equatorial Guinea': 'Medium', 'Eritrea': 'Medium',
    'Estonia': 'Low', 'Ethiopia': 'High', 'Fiji': 'Medium', 'Finland': 'Low', 'France': 'Low',
    'Gabon': 'Medium', 'Gambia': 'Medium', 'Georgia': 'Medium', 'Germany': 'Low', 'Ghana': 'Medium',
    'Gibraltar': 'Medium', 'Greece': 'Low', 'Grenada': 'Low', 'Guatemala': 'Medium',
    'Guinea': 'High', 'Guinea-Bissau': 'High', 'Guyana': 'Medium', 'Haiti': 'High',
    'Honduras': 'Low', 'Hong Kong': 'High', 'Hungary': 'Low', 'Iceland': 'Low', 'India': 'Medium',
    'Indonesia': 'Medium', 'Iran': 'Override', 'Iraq': 'High', 'Ireland': 'Medium',
    'Isle of Man': 'Low', 'Israel': 'Medium', 'Italy': 'Low', 'Ivory Coast': 'High',
    'Jamaica': 'Medium', 'Japan': 'Low', 'Jersey': 'Low', 'Jordan': 'Medium',
    'Kazakhstan': 'Medium', 'Kenya': 'High', 'Kiribati': 'Medium', 'Kosovo': 'Medium',
    'Kuwait': 'High', 'Kyrgyzstan': 'Medium', 'Laos': 'High', 'Latvia': 'Low', 'Lebanon': 'High',
    'Lesotho': 'Low', 'Liberia': 'High', 'Libya': 'High', 'Liechtenstein': 'Medium',
    'Lithuania': 'Low', 'Luxembourg': 'Medium', 'Macao': 'Low', 'Macedonia': 'Low',
    'Madagascar': 'High', 'Malawi': 'Medium', 'Malaysia': 'Low', 'Maldives': 'Medium',
    'Mali': 'Medium', 'Malta': 'Medium', 'Mauritania': 'Medium', 'Mauritius': 'Low',
    'Mexico': 'Medium', 'Moldova': 'Medium', 'Monaco': 'High', 'Mongolia': 'Medium',
    'Montenegro': 'Low', 'Morocco': 'High', 'Mozambique': 'Medium', 'Myanmar': 'Override',
    'Namibia': 'High', 'Nepal': 'High', 'Netherlands': 'Medium', 'New Zealand': 'Low',
    'Nicaragua': 'High', 'Niger': 'Medium', 'Nigeria': 'Medium', 'North Korea': 'Override',
    'Norway': 'Low', 'Oman': 'Low', 'Pakistan': 'High', 'Palau': 'High', 'Palestine': 'High',
    'Panama': 'Medium', 'Papua New Guinea': 'High', 'Paraguay': 'High', 'Peru': 'Medium',
    'Philippines': 'Medium', 'Poland': 'Low', 'Portugal': 'Low', 'Qatar': 'Medium',
    'Republic of Congo': 'High', 'Romania': 'Medium', 'Russia': 'Medium', 'Rwanda': 'Low',
    'Saint Kitts and Nevis': 'Medium', 'Saint Lucia': 'High', 'Samoa': 'High',
    'San Marino': 'Low', 'Saudi Arabia': 'Medium', 'Senegal': 'Medium', 'Serbia': 'Medium',
    'Seychelles': 'Medium', 'Sierra Leone': 'High', 'Singapore': 'Low', 'Slovakia': 'Low',
    'Slovenia': 'Low', 'Solomon Islands': 'High', 'Somalia': 'High', 'South Africa': 'Medium',
    'South Korea': 'Low', 'South Sudan': 'High', 'Spain': 'Low', 'Sri Lanka': 'Medium',
    'Sudan': 'High', 'Suriname': 'Medium', 'Sweden': 'Low', 'Switzerland': 'Medium',
    'Syria': 'High', 'Taiwan': 'Low', 'Tajikistan': 'Medium', 'Tanzania': 'Medium',
    'Thailand': 'Medium', 'Togo': 'Medium', 'Tonga': 'High', 'Trinidad and Tobago': 'Low',
    'Tunisia': 'High', 'Turkey': 'Medium', 'Turkmenistan': 'Medium', 'Tuvalu': 'Medium',
    'Uganda': 'Medium', 'Ukraine': 'High', 'United Arab Emirates': 'Low',
    'United Kingdom': 'Low', 'United States': 'Low', 'Uruguay': 'Low', 'Uzbekistan': 'Medium',
    'Vanuatu': 'Low', 'Venezuela': 'High', 'Vietnam': 'High', 'Yemen': 'High',
    'Zambia': 'Medium', 'Zimbabwe': 'High'
}


# ══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING (cached)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def train_model():
    """Train the Gradient Boosting model from historical data."""
    try:
        df = pd.read_excel(TRAINING_DATA)
    except FileNotFoundError:
        return None, None

    df = df[df['Renewal Decision'].isin(['Renewed', 'Churned'])].reset_index(drop=True)
    df['Target'] = (df['Renewal Decision'] == 'Renewed').astype(int)

    # Encode
    df['Weighted Activity Risk Encoded'] = df['Weighted Activity Risk'].map(RISK_MAP)
    df['Weighted Nationality Risk Encoded'] = df['Weighted Nationality Risk'].map(RISK_MAP)
    df['Third-party Approval Encoded'] = df['Third-party Approval Required'].map(APPROVAL_MAP)

    # Engineer features
    df['Zero Transactions Flag'] = (df['Latest Year Transactions'] == 0).astype(int)
    df['Log Latest Transactions'] = np.log1p(df['Latest Year Transactions'])
    df['Log Cumulative Transactions'] = np.log1p(df['Cumulative Transactions'])
    df['Engagement Ratio'] = np.where(
        df['Cumulative Transactions'] > 0,
        df['Latest Year Transactions'] / df['Cumulative Transactions'], 0)
    df['Transactions per Visa'] = np.where(
        df['Visa Allocation'] > 0,
        df['Latest Year Transactions'] / df['Visa Allocation'],
        df['Latest Year Transactions'])
    df['Combined Risk Score'] = df['Weighted Activity Risk Encoded'] + df['Weighted Nationality Risk Encoded']

    X = df[FEATURE_COLS]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    model = GradientBoostingClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        min_samples_leaf=15, subsample=0.9, random_state=42)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        'auc': round(roc_auc_score(y_test, y_prob), 4),
        'brier': round(brier_score_loss(y_test, y_prob), 4),
        'accuracy': round((model.predict(X_test) == y_test).mean(), 4),
        'train_size': len(X_train),
        'test_size': len(X_test)
    }

    return model, metrics


# ══════════════════════════════════════════════════════════════════════════════
# PROCESSING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def compute_weighted_nationality_risk(row, sh_nat_cols):
    """Compute weighted nationality risk from shareholder nationality columns."""
    risks = []
    for col in sh_nat_cols:
        nat = row.get(col)
        if pd.notna(nat) and str(nat).strip():
            risk = GEO_RISK.get(str(nat).strip())
            if risk and risk in RISK_MAP:
                risks.append(risk)
    if not risks:
        return 'Medium'
    if 'Override' in risks:
        return 'Override'
    avg = np.mean([RISK_MAP[r] for r in risks])
    if avg <= 1.5: return 'Low'
    elif avg <= 2.5: return 'Medium'
    else: return 'High'


def process_upload(df_input):
    """Process uploaded company data into model-ready features."""
    result = pd.DataFrame()

    result['Company Name'] = df_input['Company Name']
    result['License Issue Date'] = pd.to_datetime(df_input['License Issue Date'])
    result['License Expiry Date'] = pd.to_datetime(df_input['License Expiry Date'])
    result['Total No of Activities'] = df_input['Total No of Activities'].fillna(1).astype(int)
    result['No of Groups Opted'] = df_input['No of Groups Opted'].fillna(1).astype(int)
    result['Visa Allocation'] = df_input['Visa Allocation'].fillna(0).astype(int)
    result['Number of Shareholders'] = df_input['Number of Shareholders'].fillna(1).astype(int)

    # Third-party approval
    if 'Third-party Approval Required' in df_input.columns:
        result['Third-party Approval Encoded'] = df_input['Third-party Approval Required'].map(APPROVAL_MAP).fillna(0).astype(int)
    else:
        result['Third-party Approval Encoded'] = 0

    # Weighted Activity Risk
    if 'Weighted Activity Risk' in df_input.columns:
        result['Weighted Activity Risk Encoded'] = df_input['Weighted Activity Risk'].map(RISK_MAP).fillna(2).astype(int)
    else:
        result['Weighted Activity Risk Encoded'] = 2

    # Shareholder age columns
    sh_age_cols = [c for c in df_input.columns if 'age' in c.lower() and 'shareholder' in c.lower()]
    if sh_age_cols:
        result['Average Shareholder Age'] = df_input[sh_age_cols].apply(
            lambda row: round(row.dropna().mean(), 1) if row.dropna().any() else 42.8, axis=1)
    else:
        result['Average Shareholder Age'] = 42.8

    # Shareholder nationality columns → weighted nationality risk
    sh_nat_cols = [c for c in df_input.columns if 'nationality' in c.lower() and 'shareholder' in c.lower()]
    if sh_nat_cols:
        result['Weighted Nationality Risk Encoded'] = df_input.apply(
            lambda row: RISK_MAP[compute_weighted_nationality_risk(row, sh_nat_cols)], axis=1)
    else:
        result['Weighted Nationality Risk Encoded'] = 2

    # Year columns — find all Year/Y columns
    year_cols = []
    for i in range(1, 6):
        candidates = [c for c in df_input.columns if
                      c.strip().lower() in [f'year {i}', f'y{i}', f'year{i}',
                                            f'y {i}', f'year {i} transactions']]
        if candidates:
            year_cols.append((i, candidates[0]))

    # Determine renewal year from license issue date
    result['Renewal Year'] = result['License Issue Date'].apply(
        lambda d: max(1, int((REF_DATE - d).days / 365)) if pd.notna(d) else 1)
    # Cap at 5
    result['Renewal Year'] = result['Renewal Year'].clip(upper=5)

    # Transaction features
    for _, row_data in result.iterrows():
        idx = row_data.name
        ry = result.at[idx, 'Renewal Year']

        # Build year values array
        year_values = []
        for yr_num, yr_col in year_cols:
            val = df_input.at[idx, yr_col]
            year_values.append(int(val) if pd.notna(val) else 0)

        # Pad to 5 years
        while len(year_values) < 5:
            year_values.append(0)

        # Latest year transactions (at the renewal year)
        ry_idx = min(ry, len(year_values)) - 1
        result.at[idx, 'Latest Year Transactions'] = year_values[ry_idx]

        # Cumulative
        cum = sum(year_values[:ry])
        result.at[idx, 'Cumulative Transactions'] = cum

        # Trend
        if ry <= 1:
            result.at[idx, 'Transaction Trend'] = 0
        else:
            prev_idx = ry - 2
            result.at[idx, 'Transaction Trend'] = year_values[ry_idx] - year_values[prev_idx]

    # Ensure numeric
    for col in ['Latest Year Transactions', 'Cumulative Transactions', 'Transaction Trend']:
        result[col] = pd.to_numeric(result[col], errors='coerce').fillna(0).astype(int)

    # Engineered features
    result['Zero Transactions Flag'] = (result['Latest Year Transactions'] == 0).astype(int)
    result['Log Latest Transactions'] = np.log1p(result['Latest Year Transactions'])
    result['Log Cumulative Transactions'] = np.log1p(result['Cumulative Transactions'])
    result['Engagement Ratio'] = np.where(
        result['Cumulative Transactions'] > 0,
        result['Latest Year Transactions'] / result['Cumulative Transactions'], 0)
    result['Transactions per Visa'] = np.where(
        result['Visa Allocation'] > 0,
        result['Latest Year Transactions'] / result['Visa Allocation'],
        result['Latest Year Transactions'])
    result['Combined Risk Score'] = (
        result['Weighted Activity Risk Encoded'] + result['Weighted Nationality Risk Encoded'])

    return result


def risk_category(prob):
    if prob < 0.4: return 'High Risk'
    elif prob < 0.6: return 'Medium Risk'
    elif prob < 0.8: return 'Low Risk'
    else: return 'Very Low Risk'


def risk_color(cat):
    return {
        'High Risk': '#E74C3C',
        'Medium Risk': '#F39C12',
        'Low Risk': '#27AE60',
        'Very Low Risk': '#2E86AB'
    }.get(cat, '#888888')


def create_template():
    """Create a downloadable Excel template."""
    template_data = {
        'Company Name': ['ABC Trading LLC', 'XYZ Holdings'],
        'License Issue Date': ['2024-01-15', '2023-06-20'],
        'License Expiry Date': ['2025-01-14', '2024-06-19'],
        'Total No of Activities': [3, 5],
        'No of Groups Opted': [2, 3],
        'Weighted Activity Risk': ['Medium', 'High'],
        'Visa Allocation': [2, 4],
        'Third-party Approval Required': ['No', 'Yes'],
        'Number of Shareholders': [2, 1],
        'Shareholder 1 Age': [35, 42],
        'Shareholder 1 Nationality': ['India', 'United Kingdom'],
        'Shareholder 2 Age': [40, None],
        'Shareholder 2 Nationality': ['Egypt', None],
        'Shareholder 3 Age': [None, None],
        'Shareholder 3 Nationality': [None, None],
        'Year 1': [5, 8],
        'Year 2': [3, 6],
        'Year 3': [0, 4],
        'Year 4': [0, 0],
        'Year 5': [0, 0],
    }
    return pd.DataFrame(template_data)


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="B2C Renewal Predictor",
    page_icon="🔄",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1F4E79;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1F4E79;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #666;
        margin-top: 0.3rem;
    }
    .risk-high { color: #E74C3C; font-weight: 700; }
    .risk-medium { color: #F39C12; font-weight: 700; }
    .risk-low { color: #27AE60; font-weight: 700; }
    .risk-vlow { color: #2E86AB; font-weight: 700; }
    .stDataFrame { font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">B2C License Renewal Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Meydan Free Zone — Upload company data to get renewal probability scores</p>', unsafe_allow_html=True)

# Load model
model, metrics = train_model()

if model is None:
    st.error(f"Training data file not found: **{TRAINING_DATA}**. "
             f"Place the file in the same directory as this script and restart.")
    st.stop()

# Sidebar — Model Info
with st.sidebar:
    st.markdown("### Model Performance")
    st.metric("AUC-ROC", f"{metrics['auc']:.3f}")
    st.metric("Brier Score", f"{metrics['brier']:.3f}")
    st.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    st.divider()
    st.markdown(f"**Training:** {metrics['train_size']:,} rows")
    st.markdown(f"**Test:** {metrics['test_size']:,} rows")
    st.markdown(f"**Model:** Gradient Boosting")
    st.markdown(f"**Features:** {len(FEATURE_COLS)}")
    st.divider()
    st.markdown("### Risk Thresholds")
    st.markdown("- 🔴 **High Risk:** < 40%")
    st.markdown("- 🟡 **Medium Risk:** 40–60%")
    st.markdown("- 🟢 **Low Risk:** 60–80%")
    st.markdown("- 🔵 **Very Low Risk:** > 80%")

# Main content
tab1, tab2 = st.tabs(["📊 Score Companies", "📋 Template & Instructions"])

# ── TAB 2: Template ──
with tab2:
    st.markdown("### Input File Template")
    st.markdown("Download the template, fill in your company data, and upload it in the scoring tab.")

    template_df = create_template()
    st.dataframe(template_df, use_container_width=True, hide_index=True)

    buffer = io.BytesIO()
    template_df.to_excel(buffer, index=False, sheet_name='Companies')
    buffer.seek(0)

    st.download_button(
        label="Download Template (.xlsx)",
        data=buffer,
        file_name="B2C_Renewal_Prediction_Template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.markdown("### Column Guide")
    st.markdown("""
    | Column | Required | Format | Notes |
    |--------|----------|--------|-------|
    | Company Name | Yes | Text | For identification only, not used in model |
    | License Issue Date | Yes | Date | When the license was first issued |
    | License Expiry Date | Yes | Date | Current expiry date |
    | Total No of Activities | Yes | Integer | Number of business activities registered |
    | No of Groups Opted | Yes | Integer | Number of activity groups |
    | Weighted Activity Risk | Yes | Low / Medium / High / Override | Pre-calculated weighted risk |
    | Visa Allocation | Yes | Integer | Number of visas allocated (0 if none) |
    | Third-party Approval Required | No | Yes / No | Defaults to No if missing |
    | Number of Shareholders | Yes | Integer | Total shareholder count |
    | Shareholder N Age | Yes | Number | Age of Nth shareholder (add columns as needed) |
    | Shareholder N Nationality | Yes | Text | Country name matching the geo risk list |
    | Year 1, Year 2, ... | Yes | Integer | Transaction count per year (0 if none) |
    """)

    st.markdown("### Supported Nationalities")
    with st.expander("View all 165+ supported nationalities"):
        nat_df = pd.DataFrame({
            'Country': list(GEO_RISK.keys()),
            'Risk Rating': list(GEO_RISK.values())
        }).sort_values('Country')
        st.dataframe(nat_df, use_container_width=True, hide_index=True, height=400)

# ── TAB 1: Scoring ──
with tab1:
    uploaded_file = st.file_uploader(
        "Upload company data (.xlsx)",
        type=['xlsx'],
        help="Use the template from the instructions tab"
    )

    if uploaded_file is not None:
        try:
            df_input = pd.read_excel(uploaded_file)
            st.success(f"Loaded {len(df_input)} companies")

            # Validate required columns
            required = ['Company Name', 'License Issue Date', 'Total No of Activities',
                        'No of Groups Opted', 'Visa Allocation', 'Number of Shareholders']
            missing = [c for c in required if c not in df_input.columns]
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
                st.stop()

            # Check for year columns
            year_cols_found = [c for c in df_input.columns if
                               any(c.strip().lower() in [f'year {i}', f'y{i}', f'year{i}', f'y {i}']
                                   for i in range(1, 6))]
            if not year_cols_found:
                st.error("No Year/Transaction columns found. Expected: Year 1, Year 2, ... or Y1, Y2, ...")
                st.stop()

            # Process
            with st.spinner("Processing and scoring companies..."):
                processed = process_upload(df_input)

                # Score
                X_score = processed[FEATURE_COLS]
                processed['Renewal Probability'] = model.predict_proba(X_score)[:, 1]
                processed['Renewal Probability'] = processed['Renewal Probability'].round(4)
                processed['Risk Category'] = processed['Renewal Probability'].apply(risk_category)

            # ── Summary metrics ──
            st.markdown("---")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value">{len(processed)}</div>
                    <div class="metric-label">Companies Scored</div>
                </div>""", unsafe_allow_html=True)
            with col2:
                high_risk = (processed['Risk Category'] == 'High Risk').sum()
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value risk-high">{high_risk}</div>
                    <div class="metric-label">High Risk</div>
                </div>""", unsafe_allow_html=True)
            with col3:
                med_risk = (processed['Risk Category'] == 'Medium Risk').sum()
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value risk-medium">{med_risk}</div>
                    <div class="metric-label">Medium Risk</div>
                </div>""", unsafe_allow_html=True)
            with col4:
                low_risk = (processed['Risk Category'] == 'Low Risk').sum()
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value risk-low">{low_risk}</div>
                    <div class="metric-label">Low Risk</div>
                </div>""", unsafe_allow_html=True)
            with col5:
                vlow_risk = (processed['Risk Category'] == 'Very Low Risk').sum()
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value risk-vlow">{vlow_risk}</div>
                    <div class="metric-label">Very Low Risk</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("---")

            # ── Results table ──
            display_cols = ['Company Name', 'Renewal Probability', 'Risk Category',
                            'Renewal Year', 'Latest Year Transactions',
                            'Cumulative Transactions', 'Transaction Trend',
                            'Average Shareholder Age', 'Visa Allocation',
                            'Zero Transactions Flag']

            display_df = processed[display_cols].sort_values(
                'Renewal Probability', ascending=True).reset_index(drop=True)

            st.markdown("### Scored Companies")
            st.markdown("Sorted by renewal probability (highest risk first)")

            # Color-code the risk category
            def highlight_risk(val):
                colors = {
                    'High Risk': 'background-color: #FDEDEC; color: #E74C3C; font-weight: bold',
                    'Medium Risk': 'background-color: #FEF9E7; color: #F39C12; font-weight: bold',
                    'Low Risk': 'background-color: #EAFAF1; color: #27AE60; font-weight: bold',
                    'Very Low Risk': 'background-color: #EBF5FB; color: #2E86AB; font-weight: bold'
                }
                return colors.get(val, '')

            styled = display_df.style.applymap(
                highlight_risk, subset=['Risk Category']
            ).format({'Renewal Probability': '{:.1%}'})

            st.dataframe(styled, use_container_width=True, hide_index=True, height=500)

            # ── Risk distribution chart ──
            st.markdown("### Risk Distribution")
            col_chart, col_stats = st.columns([2, 1])

            with col_chart:
                risk_counts = processed['Risk Category'].value_counts()
                risk_order = ['High Risk', 'Medium Risk', 'Low Risk', 'Very Low Risk']
                risk_counts = risk_counts.reindex(risk_order, fill_value=0)

                chart_df = pd.DataFrame({
                    'Risk Category': risk_counts.index,
                    'Count': risk_counts.values
                })
                st.bar_chart(chart_df.set_index('Risk Category'), height=300)

            with col_stats:
                st.markdown("**Renewal Probability Statistics**")
                stats = processed['Renewal Probability'].describe()
                st.markdown(f"""
                - **Mean:** {stats['mean']:.1%}
                - **Median:** {stats['50%']:.1%}
                - **Std Dev:** {stats['std']:.1%}
                - **Min:** {stats['min']:.1%}
                - **Max:** {stats['max']:.1%}
                """)

            # ── Download results ──
            st.markdown("---")

            output_df = processed[['Company Name', 'License Issue Date', 'License Expiry Date',
                                    'Renewal Year', 'Visa Allocation', 'Total No of Activities',
                                    'No of Groups Opted', 'Number of Shareholders',
                                    'Average Shareholder Age',
                                    'Latest Year Transactions', 'Cumulative Transactions',
                                    'Transaction Trend', 'Engagement Ratio', 'Transactions per Visa',
                                    'Zero Transactions Flag',
                                    'Renewal Probability', 'Risk Category']].copy()
            output_df = output_df.sort_values('Renewal Probability', ascending=True).reset_index(drop=True)

            out_buffer = io.BytesIO()
            output_df.to_excel(out_buffer, index=False, sheet_name='Renewal Scores')
            out_buffer.seek(0)

            st.download_button(
                label="Download Scored Results (.xlsx)",
                data=out_buffer,
                file_name="B2C_Renewal_Scores.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.exception(e)
    else:
        st.info("Upload an Excel file to get started. Download the template from the instructions tab.")
