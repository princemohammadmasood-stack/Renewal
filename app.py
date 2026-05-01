"""
B2C License Renewal Prediction App
"""
import streamlit as st
import pandas as pd
import numpy as np
import io
import pathlib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss

REF_DATE = pd.Timestamp('2026-01-01')
APP_DIR = pathlib.Path(__file__).parent.resolve()
TRAINING_DATA = str(APP_DIR / "B2C_Renewal_Decision_Data.xlsx")
ACTIVITY_FILE = str(APP_DIR / "Activity List.xlsx")
GEO_RISK_FILE = str(APP_DIR / "Geographical Risk Rating List.xlsx")

RISK_MAP = {'Low': 1, 'Medium': 2, 'High': 3, 'Override': 4}
APPROVAL_MAP = {'No': 0, 'Yes': 1}
RISK_LABEL = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Override'}

FEATURE_COLS = [
    'Visa Allocation', 'Total No of Activities', 'No of Groups Opted',
    'Number of Shareholders', 'Average Shareholder Age', 'Renewal Year',
    'Latest Year Transactions', 'Cumulative Transactions', 'Transaction Trend',
    'Weighted Activity Risk Encoded', 'Weighted Nationality Risk Encoded',
    'Third-party Approval Encoded', 'Zero Transactions Flag',
    'Log Latest Transactions', 'Log Cumulative Transactions',
    'Engagement Ratio', 'Transactions per Visa', 'Combined Risk Score']


@st.cache_data
def load_activity_reference():
    try:
        df = pd.read_excel(ACTIVITY_FILE)
        col_map = {}
        for c in df.columns:
            cl = c.strip().lower()
            if cl == 'code':
                col_map[c] = 'Code'
            elif 'activity name' in cl or cl == 'activity':
                col_map[c] = 'Activity Name'
            elif cl == 'group':
                col_map[c] = 'Group'
            elif cl == 'category':
                col_map[c] = 'Category'
            elif 'risk' in cl and 'rating' in cl:
                col_map[c] = 'Risk Rating'
            elif 'third' in cl and 'party' in cl:
                col_map[c] = 'Third Party'
        df.rename(columns=col_map, inplace=True)
        df = df.dropna(subset=['Code', 'Activity Name'])
        df['Code'] = df['Code'].astype(str).str.strip()
        df['Activity Name'] = df['Activity Name'].astype(str).str.strip()
        df['display'] = df['Code'] + ' - ' + df['Activity Name']
        df['Risk Rating'] = df['Risk Rating'].astype(str).str.strip().replace({'nan': 'Low', '': 'Low'})
        df['Group'] = df['Group'].astype(str).str.strip()
        if 'Third Party' in df.columns:
            df['Third Party'] = df['Third Party'].astype(str).str.strip()
        else:
            df['Third Party'] = ''
        if 'Category' in df.columns:
            df['Category'] = df['Category'].astype(str).str.strip()
        return df
    except Exception as e:
        st.error(f"Error loading Activity List: {e}")
        return None


@st.cache_data
def load_geo_risk():
    try:
        df = pd.read_excel(GEO_RISK_FILE)
        col_map = {}
        for c in df.columns:
            cl = c.strip().lower()
            if 'country' in cl:
                col_map[c] = 'Country'
            elif 'risk' in cl or 'geo' in cl:
                col_map[c] = 'Risk'
        df.rename(columns=col_map, inplace=True)
        df = df.dropna(subset=['Country', 'Risk'])
        df = df[df['Country'] != 'No filters applied']
        df['Country'] = df['Country'].astype(str).str.strip()
        df['Risk'] = df['Risk'].astype(str).str.strip()
        return dict(zip(df['Country'], df['Risk']))
    except Exception as e:
        st.error(f"Error loading Geo Risk: {e}")
        return None


@st.cache_resource
def train_model():
    try:
        df = pd.read_excel(TRAINING_DATA)
        df = df[df['Renewal Decision'].isin(['Renewed', 'Churned'])].reset_index(drop=True)
        df['Target'] = (df['Renewal Decision'] == 'Renewed').astype(int)
        df['Weighted Activity Risk Encoded'] = df['Weighted Activity Risk'].map(RISK_MAP)
        df['Weighted Nationality Risk Encoded'] = df['Weighted Nationality Risk'].map(RISK_MAP)
        df['Third-party Approval Encoded'] = df['Third-party Approval Required'].map(APPROVAL_MAP)
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        mdl = GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            min_samples_leaf=15, subsample=0.9, random_state=42)
        mdl.fit(X_train, y_train)
        yp = mdl.predict_proba(X_test)[:, 1]
        m = {
            'auc': round(roc_auc_score(y_test, yp), 4),
            'brier': round(brier_score_loss(y_test, yp), 4),
            'accuracy': round((mdl.predict(X_test) == y_test).mean(), 4),
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
        return mdl, m
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None, None


def compute_weighted_risk(risk_list):
    risks = [r for r in risk_list if r in RISK_MAP]
    if not risks:
        return 'Medium'
    if 'Override' in risks:
        return 'Override'
    avg = np.mean([RISK_MAP[r] for r in risks])
    if avg <= 1.5:
        return 'Low'
    elif avg <= 2.5:
        return 'Medium'
    else:
        return 'High'


def derive_activity_features(activity_codes, act_ref):
    if act_ref is None or not activity_codes:
        return 0, 0, 'Medium', 'No'
    matched = act_ref[act_ref['Code'].isin(activity_codes)]
    total_activities = len(matched) if len(matched) > 0 else len(activity_codes)
    groups = matched['Group'].dropna().unique()
    groups = [g for g in groups if g and g != 'nan']
    no_groups = len(groups) if groups else 1
    risk_list = matched['Risk Rating'].tolist()
    weighted_risk = compute_weighted_risk(risk_list) if risk_list else 'Medium'
    third_party = 'No'
    if 'Third Party' in matched.columns:
        tp_vals = matched['Third Party'].dropna().tolist()
        tp_vals = [t for t in tp_vals if t and t != 'nan' and t.strip() != '' and t.strip() != 'N/A']
        if tp_vals:
            third_party = 'Yes'
    return total_activities, no_groups, weighted_risk, third_party


def parse_activity_string(activity_str, act_ref):
    if pd.isna(activity_str) or not str(activity_str).strip():
        return []
    parts = str(activity_str).split(',')
    codes = []
    for p in parts:
        p = p.strip()
        if ' - ' in p:
            code = p.split(' - ')[0].strip()
            codes.append(code)
        elif p:
            codes.append(p.strip())
    return codes


def add_engineered(df):
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
    return df


def risk_category(p):
    if p < 0.4:
        return 'High Risk'
    elif p < 0.6:
        return 'Medium Risk'
    elif p < 0.8:
        return 'Low Risk'
    else:
        return 'Very Low Risk'


def risk_emoji(c):
    return {'High Risk': '🔴', 'Medium Risk': '🟡', 'Low Risk': '🟢', 'Very Low Risk': '🔵'}.get(c, '⚪')


def risk_color(c):
    return {'High Risk': '#E74C3C', 'Medium Risk': '#F39C12', 'Low Risk': '#27AE60', 'Very Low Risk': '#2E86AB'}.get(c, '#888')


def process_manual(d, act_ref, geo_risk):
    total_act, no_groups, w_act_risk, third_party = derive_activity_features(d['activity_codes'], act_ref)
    ages = [s['age'] for s in d['shareholders'] if s['age'] > 0]
    nats = [s['nationality'] for s in d['shareholders']]
    nat_risks = [geo_risk.get(n, 'Medium') for n in nats if n]
    w_nat_risk = compute_weighted_risk(nat_risks)
    ry = max(1, min(5, int((REF_DATE - pd.Timestamp(d['issue_date'])).days / 365)))
    yv = d['year_transactions'] + [0] * (5 - len(d['year_transactions']))
    ri = ry - 1
    lt = yv[ri]
    cum = sum(yv[:ry])
    trend = 0 if ry <= 1 else yv[ri] - yv[ri - 1]
    row = pd.DataFrame([{
        'Company Name': d['company_name'],
        'License Issue Date': d['issue_date'],
        'License Expiry Date': d['expiry_date'],
        'Total No of Activities': total_act,
        'No of Groups Opted': no_groups,
        'Visa Allocation': d['visa_allocation'],
        'Number of Shareholders': d['num_shareholders'],
        'Third-party Approval Encoded': APPROVAL_MAP.get(third_party, 0),
        'Weighted Activity Risk Encoded': RISK_MAP.get(w_act_risk, 2),
        'Average Shareholder Age': round(np.mean(ages), 1) if ages else 42.8,
        'Weighted Nationality Risk Encoded': RISK_MAP.get(w_nat_risk, 2),
        'Renewal Year': ry,
        'Latest Year Transactions': lt,
        'Cumulative Transactions': cum,
        'Transaction Trend': trend,
        '_w_act_risk_label': w_act_risk,
        '_w_nat_risk_label': w_nat_risk,
        '_third_party': third_party
    }])
    return add_engineered(row)


def process_upload(df_in, act_ref, geo_risk):
    r = pd.DataFrame()
    r['Company Name'] = df_in['Company Name']
    r['License Issue Date'] = pd.to_datetime(df_in['License Issue Date'])
    if 'License Expiry Date' in df_in.columns:
        r['License Expiry Date'] = pd.to_datetime(df_in['License Expiry Date'])

    if 'Business Activity' in df_in.columns and act_ref is not None:
        act_features = df_in['Business Activity'].apply(
            lambda x: derive_activity_features(parse_activity_string(x, act_ref), act_ref))
        r['Total No of Activities'] = act_features.apply(lambda x: x[0])
        r['No of Groups Opted'] = act_features.apply(lambda x: x[1])
        r['Weighted Activity Risk Encoded'] = act_features.apply(lambda x: RISK_MAP.get(x[2], 2))
        r['Third-party Approval Encoded'] = act_features.apply(lambda x: APPROVAL_MAP.get(x[3], 0))
    else:
        r['Total No of Activities'] = df_in.get('Total No of Activities', pd.Series([1] * len(df_in))).fillna(1).astype(int)
        r['No of Groups Opted'] = df_in.get('No of Groups Opted', pd.Series([1] * len(df_in))).fillna(1).astype(int)
        if 'Weighted Activity Risk' in df_in.columns:
            r['Weighted Activity Risk Encoded'] = df_in['Weighted Activity Risk'].map(RISK_MAP).fillna(2).astype(int)
        else:
            r['Weighted Activity Risk Encoded'] = 2
        if 'Third-party Approval Required' in df_in.columns:
            r['Third-party Approval Encoded'] = df_in['Third-party Approval Required'].map(APPROVAL_MAP).fillna(0).astype(int)
        else:
            r['Third-party Approval Encoded'] = 0

    r['Visa Allocation'] = df_in.get('Visa Allocation', pd.Series([0] * len(df_in))).fillna(0).astype(int)
    r['Number of Shareholders'] = df_in.get('Number of Shareholders', pd.Series([1] * len(df_in))).fillna(1).astype(int)

    sac = [c for c in df_in.columns if 'age' in c.lower() and 'shareholder' in c.lower()]
    if sac:
        r['Average Shareholder Age'] = df_in[sac].apply(
            lambda x: round(x.dropna().mean(), 1) if x.dropna().any() else 42.8, axis=1)
    else:
        r['Average Shareholder Age'] = 42.8

    snc = [c for c in df_in.columns if 'nationality' in c.lower() and 'shareholder' in c.lower()]
    if snc and geo_risk:
        r['Weighted Nationality Risk Encoded'] = df_in.apply(
            lambda row: RISK_MAP[compute_weighted_risk(
                [geo_risk.get(str(row.get(c, '')).strip(), 'Medium') for c in snc
                 if pd.notna(row.get(c)) and str(row.get(c)).strip()])], axis=1)
    else:
        r['Weighted Nationality Risk Encoded'] = 2

    yc = []
    for i in range(1, 6):
        cands = [c for c in df_in.columns if c.strip().lower() in
                 [f'year {i}', f'y{i}', f'year{i}', f'y {i}', f'year {i} transactions']]
        if cands:
            yc.append((i, cands[0]))

    r['Renewal Year'] = r['License Issue Date'].apply(
        lambda d: max(1, min(5, int((REF_DATE - d).days / 365))) if pd.notna(d) else 1)

    for _, rd in r.iterrows():
        idx = rd.name
        ry = r.at[idx, 'Renewal Year']
        yv = [int(df_in.at[idx, c]) if pd.notna(df_in.at[idx, c]) else 0 for _, c in yc]
        yv += [0] * (5 - len(yv))
        ri = min(ry, len(yv)) - 1
        r.at[idx, 'Latest Year Transactions'] = yv[ri]
        r.at[idx, 'Cumulative Transactions'] = sum(yv[:ry])
        r.at[idx, 'Transaction Trend'] = 0 if ry <= 1 else yv[ri] - yv[ri - 1]

    for c in ['Latest Year Transactions', 'Cumulative Transactions', 'Transaction Trend']:
        r[c] = pd.to_numeric(r[c], errors='coerce').fillna(0).astype(int)
    return add_engineered(r)


# ══════════════════════════════════════════════════════════════════════════════
# APP UI
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="B2C Renewal Predictor", page_icon="🔄", layout="wide")

st.markdown("""<style>
.main-header{font-size:2.2rem;font-weight:700;color:#1F4E79;margin-bottom:0}
.sub-header{font-size:1rem;color:#666;margin-top:0;margin-bottom:2rem}
.metric-card{background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:1.2rem;text-align:center}
.metric-value{font-size:1.8rem;font-weight:700;color:#1F4E79}
.metric-label{font-size:.85rem;color:#666;margin-top:.3rem}
.risk-high{color:#E74C3C;font-weight:700}
.risk-medium{color:#F39C12;font-weight:700}
.risk-low{color:#27AE60;font-weight:700}
.risk-vlow{color:#2E86AB;font-weight:700}
.result-box{padding:2rem;border-radius:12px;text-align:center;margin:1rem 0}
.score-big{font-size:4rem;font-weight:800;line-height:1}
.score-label{font-size:1.2rem;font-weight:600;margin-top:.5rem}
.factor-card{background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:.8rem 1rem;margin:.3rem 0}
.factor-title{font-size:.8rem;color:#888}
.factor-value{font-size:1.1rem;font-weight:600;color:#1F4E79}
.derived-box{background:#EBF5FB;border:1px solid #2E86AB;border-radius:8px;padding:1rem;margin:.5rem 0}
</style>""", unsafe_allow_html=True)

st.markdown('<p class="main-header">B2C License Renewal Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Meydan Free Zone — Enter company details or upload in bulk to get renewal scores</p>', unsafe_allow_html=True)

# Load everything
model, metrics = train_model()
act_ref = load_activity_reference()
geo_risk = load_geo_risk()

# Validation
missing_files = []
if model is None:
    missing_files.append("Training Data")
if act_ref is None:
    missing_files.append("Activity List")
if geo_risk is None:
    missing_files.append("Geo Risk List")
if missing_files:
    st.warning(f"Could not load: **{', '.join(missing_files)}**. Check the error messages above.")
    st.stop()

NATIONALITY_LIST = sorted(geo_risk.keys())

# Sidebar
with st.sidebar:
    st.markdown("### Model Performance")
    st.metric("AUC-ROC", f"{metrics['auc']:.3f}")
    st.metric("Brier Score", f"{metrics['brier']:.3f}")
    st.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    st.divider()
    st.markdown(f"**Training:** {metrics['train_size']:,} rows")
    st.markdown(f"**Activities loaded:** {len(act_ref):,}")
    st.markdown(f"**Countries loaded:** {len(geo_risk):,}")
    st.divider()
    st.markdown("### Risk Thresholds")
    st.markdown("- 🔴 **High Risk:** < 40%")
    st.markdown("- 🟡 **Medium Risk:** 40–60%")
    st.markdown("- 🟢 **Low Risk:** 60–80%")
    st.markdown("- 🔵 **Very Low Risk:** > 80%")

tab1, tab2, tab3 = st.tabs(["✍️ Manual Input", "📊 Bulk Upload", "📋 Guide"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MANUAL INPUT
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Company Details")
    ca, cb = st.columns(2)
    with ca:
        company_name = st.text_input("Company Name", placeholder="e.g. ABC Trading LLC")
        issue_date = st.date_input("License Issue Date", value=pd.Timestamp('2024-01-01'))
        expiry_date = st.date_input("License Expiry Date", value=pd.Timestamp('2025-12-31'))
        visa_allocation = st.number_input("Visa Allocation", min_value=0, max_value=600, value=2)
    with cb:
        num_shareholders = st.number_input("Number of Shareholders", min_value=1, max_value=11, value=1)
        st.markdown("**Yearly Transactions**")
        yr_cols_input = st.columns(5)
        year_txns = []
        for i, yc_col in enumerate(yr_cols_input):
            with yc_col:
                v = st.number_input(f"Y{i+1}", min_value=0, max_value=500, value=0, key=f"yr_{i}")
                year_txns.append(v)

    st.markdown("---")
    st.markdown("### Business Activities")
    st.caption("Search and select activities. Risk, groups, and third-party approval are auto-calculated.")
    activity_options = act_ref['display'].tolist()
    selected_activities = st.multiselect(
        "Select Activities (search by code or name)",
        options=activity_options,
        default=None,
        placeholder="Type to search... e.g. 'General Trading' or '4690'")

    selected_codes = [a.split(' - ')[0].strip() for a in selected_activities]
    total_act, no_groups, w_act_risk, third_party = derive_activity_features(selected_codes, act_ref)

    if selected_activities:
        st.markdown(f"""<div class="derived-box">
            <strong>Auto-derived:</strong> Activities: <strong>{total_act}</strong> |
            Groups: <strong>{no_groups}</strong> |
            Activity Risk: <strong>{w_act_risk}</strong> |
            Third-party: <strong>{third_party}</strong>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Shareholder Details")
    shareholders = []
    for row_start in range(0, num_shareholders, 4):
        row_end = min(row_start + 4, num_shareholders)
        cols = st.columns(row_end - row_start)
        for i, col in zip(range(row_start, row_end), cols):
            with col:
                st.markdown(f"**Shareholder {i+1}**")
                age = st.number_input("Age", min_value=18, max_value=100, value=35, key=f"sh_age_{i}")
                nationality = st.selectbox("Nationality", NATIONALITY_LIST,
                    index=NATIONALITY_LIST.index('United Arab Emirates'), key=f"sh_nat_{i}")
                shareholders.append({'age': age, 'nationality': nationality})

    st.markdown("---")
    if st.button("🔍 Calculate Renewal Score", type="primary", use_container_width=True):
        if not company_name.strip():
            st.warning("Please enter a company name.")
        elif not selected_activities:
            st.warning("Please select at least one business activity.")
        else:
            data = {
                'company_name': company_name, 'issue_date': issue_date,
                'expiry_date': expiry_date, 'visa_allocation': visa_allocation,
                'num_shareholders': num_shareholders, 'activity_codes': selected_codes,
                'shareholders': shareholders, 'year_transactions': year_txns}

            processed = process_manual(data, act_ref, geo_risk)
            prob = model.predict_proba(processed[FEATURE_COLS])[0, 1]
            risk = risk_category(prob)
            emoji = risk_emoji(risk)
            color = risk_color(risk)

            st.markdown("---")
            rc1, rc2 = st.columns([1, 2])
            with rc1:
                st.markdown(f"""<div class="result-box" style="background:{color}15;border:2px solid {color};">
                    <div style="font-size:1rem;color:#666;">Renewal Probability</div>
                    <div class="score-big" style="color:{color};">{prob:.1%}</div>
                    <div class="score-label" style="color:{color};">{emoji} {risk}</div>
                </div>""", unsafe_allow_html=True)
                st.markdown(f"""<div class="result-box" style="background:#F8FAFC;border:1px solid #E2E8F0;">
                    <div style="font-size:.9rem;color:#666;">Company</div>
                    <div style="font-size:1.3rem;font-weight:700;color:#1F4E79;">{company_name}</div>
                </div>""", unsafe_allow_html=True)
            with rc2:
                st.markdown("**Key Factors**")
                vals = processed.iloc[0]
                f1, f2, f3 = st.columns(3)
                with f1:
                    st.markdown(f'<div class="factor-card"><div class="factor-title">Renewal Year</div><div class="factor-value">{int(vals["Renewal Year"])}</div></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="factor-card"><div class="factor-title">Latest Year Txns</div><div class="factor-value">{int(vals["Latest Year Transactions"])}</div></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="factor-card"><div class="factor-title">Cumulative Txns</div><div class="factor-value">{int(vals["Cumulative Transactions"])}</div></div>', unsafe_allow_html=True)
                with f2:
                    st.markdown(f'<div class="factor-card"><div class="factor-title">Engagement Ratio</div><div class="factor-value">{vals["Engagement Ratio"]:.2f}</div></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="factor-card"><div class="factor-title">Txns per Visa</div><div class="factor-value">{vals["Transactions per Visa"]:.1f}</div></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="factor-card"><div class="factor-title">Trend</div><div class="factor-value">{int(vals["Transaction Trend"]):+d}</div></div>', unsafe_allow_html=True)
                with f3:
                    st.markdown(f'<div class="factor-card"><div class="factor-title">Activity Risk</div><div class="factor-value">{vals.get("_w_act_risk_label", w_act_risk)}</div></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="factor-card"><div class="factor-title">Nationality Risk</div><div class="factor-value">{vals.get("_w_nat_risk_label", "Medium")}</div></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="factor-card"><div class="factor-title">Zero Transactions</div><div class="factor-value">{"Yes" if vals["Zero Transactions Flag"]==1 else "No"}</div></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BULK UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Upload Company Data")
    st.caption("If your file has a 'Business Activity' column, the app auto-derives risk and groups.")
    uploaded = st.file_uploader("Upload Excel (.xlsx)", type=['xlsx'], help="See the Guide tab for format")

    if uploaded is not None:
        try:
            df_in = pd.read_excel(uploaded)
            st.success(f"Loaded {len(df_in)} companies")

            required = ['Company Name', 'License Issue Date']
            missing = [c for c in required if c not in df_in.columns]
            if missing:
                st.error(f"Missing columns: {', '.join(missing)}")
                st.stop()

            ycf = [c for c in df_in.columns if any(c.strip().lower() in
                   [f'year {i}', f'y{i}', f'year{i}'] for i in range(1, 6))]
            if not ycf:
                st.error("No Year columns found.")
                st.stop()

            with st.spinner("Scoring..."):
                proc = process_upload(df_in, act_ref, geo_risk)
                proc['Renewal Probability'] = model.predict_proba(proc[FEATURE_COLS])[:, 1].round(4)
                proc['Risk Category'] = proc['Renewal Probability'].apply(risk_category)

            st.markdown("---")
            cs = st.columns(5)
            cnts = {
                't': len(proc),
                'h': (proc['Risk Category'] == 'High Risk').sum(),
                'm': (proc['Risk Category'] == 'Medium Risk').sum(),
                'l': (proc['Risk Category'] == 'Low Risk').sum(),
                'v': (proc['Risk Category'] == 'Very Low Risk').sum()
            }
            labels = [
                ('t', 'Companies', 'metric-value'),
                ('h', 'High Risk', 'metric-value risk-high'),
                ('m', 'Medium Risk', 'metric-value risk-medium'),
                ('l', 'Low Risk', 'metric-value risk-low'),
                ('v', 'Very Low Risk', 'metric-value risk-vlow')
            ]
            for col, (k, lbl, cls) in zip(cs, labels):
                with col:
                    st.markdown(f'<div class="metric-card"><div class="{cls}">{cnts[k]}</div><div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)

            st.markdown("---")
            display_cols = ['Company Name', 'Renewal Probability', 'Risk Category', 'Renewal Year',
                            'Latest Year Transactions', 'Cumulative Transactions', 'Transaction Trend',
                            'Average Shareholder Age', 'Visa Allocation', 'Zero Transactions Flag']
            display_df = proc[[c for c in display_cols if c in proc.columns]].sort_values(
                'Renewal Probability', ascending=True).reset_index(drop=True)

            st.markdown("### Scored Companies")

            def highlight_risk(val):
                return {
                    'High Risk': 'background-color:#FDEDEC;color:#E74C3C;font-weight:bold',
                    'Medium Risk': 'background-color:#FEF9E7;color:#F39C12;font-weight:bold',
                    'Low Risk': 'background-color:#EAFAF1;color:#27AE60;font-weight:bold',
                    'Very Low Risk': 'background-color:#EBF5FB;color:#2E86AB;font-weight:bold'
                }.get(val, '')

            styled = display_df.style.applymap(highlight_risk, subset=['Risk Category']).format(
                {'Renewal Probability': '{:.1%}'})
            st.dataframe(styled, use_container_width=True, hide_index=True, height=500)

            st.markdown("---")
            out_cols = [c for c in [
                'Company Name', 'License Issue Date', 'License Expiry Date',
                'Renewal Year', 'Visa Allocation', 'Total No of Activities', 'No of Groups Opted',
                'Number of Shareholders', 'Average Shareholder Age',
                'Latest Year Transactions', 'Cumulative Transactions', 'Transaction Trend',
                'Engagement Ratio', 'Transactions per Visa', 'Zero Transactions Flag',
                'Renewal Probability', 'Risk Category'] if c in proc.columns]
            output_df = proc[out_cols].sort_values('Renewal Probability').reset_index(drop=True)
            buf = io.BytesIO()
            output_df.to_excel(buf, index=False, sheet_name='Renewal Scores')
            buf.seek(0)
            st.download_button("Download Scored Results (.xlsx)", data=buf,
                              file_name="B2C_Renewal_Scores.xlsx",
                              mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                              type="primary")
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.exception(e)
    else:
        st.info("Upload an Excel file to score multiple companies at once.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — GUIDE
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Manual Input")
    st.markdown("Select business activities from the searchable dropdown. The app auto-derives Total Activities, Groups Opted, Weighted Activity Risk, and Third-party Approval.")

    st.markdown("### Bulk Upload Format")
    st.markdown("""
| Column | Required | Notes |
|--------|----------|-------|
| Company Name | Yes | Identifier |
| License Issue Date | Yes | Date |
| License Expiry Date | Optional | Date |
| Business Activity | Optional | Comma-separated codes |
| Visa Allocation | Optional | Integer, defaults to 0 |
| Number of Shareholders | Optional | Integer, defaults to 1 |
| Shareholder 1 Age | Optional | Years |
| Shareholder 1 Nationality | Optional | Country name |
| Year 1, Year 2, ... | Yes | Transaction counts |
    """)

    st.markdown("### Activity Reference")
    with st.expander(f"View all {len(act_ref)} activities"):
        display_act = act_ref[['Code', 'Activity Name', 'Category', 'Group', 'Risk Rating']].copy()
        st.dataframe(display_act, use_container_width=True, hide_index=True, height=400)

    st.markdown("### Nationality Risk Ratings")
    with st.expander(f"View all {len(geo_risk)} countries"):
        nat_df = pd.DataFrame({'Country': list(geo_risk.keys()), 'Risk': list(geo_risk.values())}).sort_values('Country')
        st.dataframe(nat_df, use_container_width=True, hide_index=True, height=400)
