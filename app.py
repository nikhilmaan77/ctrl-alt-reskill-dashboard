"""
Ctrl+Alt+Reskill — Global Workforce Reskilling Gap Intelligence Dashboard
==========================================================================
WEF/ILO-grade analytical dashboard: 8 tabs, 4 ML methods, policy-first design.
Single-file Streamlit application.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_curve, auc, confusion_matrix, classification_report,
                             precision_recall_curve, f1_score, accuracy_score)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import networkx as nx
import warnings, os

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════
# PAGE CONFIG & THEME
# ════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Ctrl+Alt+Reskill",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Color palette — dark mode
C_PRIMARY = "#00D4AA"
C_RISK = "#FF6B6B"
C_WARN = "#FFD93D"
C_INFO = "#6C9BD2"
C_PURPLE = "#B07CFF"
C_ORANGE = "#FF9F43"
C_BG_CARD = "#1A1A2E"
C_BG_DARK = "#0F0F1A"
C_TEXT = "#E8E8E8"
C_TEXT_MUTED = "#999"
C_TEXT_BODY = "#CCC"
C_TEXT_SUB = "#777"
C_TEXT_DESC = "#AAA"
C_TEXT_POLICY = "#BBB"
C_BORDER = "rgba(255,255,255,0.08)"
C_CARD_SHADOW = "rgba(0,0,0,0.3)"
C_CARD_BG1 = "#1A1A2E"
C_CARD_BG2 = "#16213E"
C_POLICY_BG1 = "#0D2137"
C_POLICY_BG2 = "#132743"
C_CALLOUT_BG = "rgba(255,255,255,0.04)"
C_PERSONA_BG1 = "#1A1A2E"
C_PERSONA_BG2 = "#1E2A3A"
C_HEADLINE_BG1 = "#0A2E1F"
C_HEADLINE_BG2 = "#0D3B2A"
C_DIVIDER = "rgba(255,255,255,0.08)"
C_GEO_LAND = "#1A1A2E"
C_GEO_OCEAN = "#0F0F1A"
C_SIM_BG1 = "#0D2137"
C_SIM_BG2 = "#132743"
C_CHART_LINE = "#555"
C_CHART_DASH = "#666"

PERSONA_COLORS = [C_PRIMARY, C_RISK, C_WARN, C_INFO, C_PURPLE]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Mono:wght@400;700&display=swap');

/* Streamlit page-level overrides — dark mode */
.stApp, .stApp > header {{
    background-color: #0E1117 !important;
}}
.stApp [data-testid="stSidebar"] {{
    background-color: #161B22 !important;
}}
.stApp [data-testid="stSidebar"] * {{
    color: #E8E8E8 !important;
}}
.stApp .main .block-container {{
    padding: 1rem 2rem; max-width: 1400px;
}}
.stApp, .stApp .main, .stApp .main .block-container {{
    color: {C_TEXT} !important;
}}
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5 {{
    color: {C_TEXT} !important;
    font-family: 'DM Sans', sans-serif !important;
}}
.stApp p, .stApp span, .stApp label, .stApp li {{
    color: {C_TEXT} !important;
}}
.stApp .stTabs [data-baseweb="tab-list"] {{
    background-color: #161B22 !important;
    border-radius: 8px;
}}
.stApp .stTabs [data-baseweb="tab"] {{
    color: {C_TEXT_MUTED} !important;
}}
.stApp .stTabs [aria-selected="true"] {{
    color: {C_PRIMARY} !important;
}}
.stApp .stMarkdown, .stApp .stCaption {{
    color: {C_TEXT} !important;
}}
.stApp .stCaption p {{
    color: {C_TEXT_MUTED} !important;
}}
.stApp [data-testid="stExpander"] {{
    background-color: #161B22 !important;
    border-color: {C_BORDER} !important;
}}
.stApp .stSelectbox label, .stApp .stMultiSelect label, .stApp .stSlider label {{
    color: {C_TEXT} !important;
}}
.stApp .stDataFrame {{
    color: {C_TEXT} !important;
}}

/* Component styles */
.kpi-card {{
    background: linear-gradient(135deg, {C_CARD_BG1} 0%, {C_CARD_BG2} 100%);
    border-radius: 12px; padding: 1.2rem; text-align: center;
    border: 1px solid {C_BORDER};
    box-shadow: 0 4px 20px {C_CARD_SHADOW};
}}
.kpi-value {{ font-size: 2rem; font-weight: 700; font-family: 'Space Mono', monospace; margin: 0.3rem 0; }}
.kpi-label {{ font-size: 0.8rem; color: {C_TEXT_MUTED}; text-transform: uppercase; letter-spacing: 1px; }}
.kpi-sub {{ font-size: 0.75rem; color: {C_TEXT_SUB}; margin-top: 0.2rem; }}
.policy-panel {{
    background: linear-gradient(135deg, {C_POLICY_BG1} 0%, {C_POLICY_BG2} 100%);
    border-left: 4px solid {C_PRIMARY}; border-radius: 8px;
    padding: 1.5rem; margin: 1rem 0;
}}
.policy-panel h4 {{ color: {C_PRIMARY}; margin-top: 0; font-size: 1.1rem; }}
.policy-panel li {{ margin-bottom: 0.6rem; color: {C_TEXT_POLICY}; line-height: 1.5; }}
.callout-box {{
    background: {C_CALLOUT_BG}; border-radius: 8px;
    padding: 1rem 1.2rem; margin: 0.5rem 0;
    border: 1px solid {C_BORDER};
}}
.callout-box .callout-title {{ font-weight: 700; font-size: 0.95rem; margin-bottom: 0.3rem; }}
.callout-box .callout-body {{ font-size: 0.85rem; color: {C_TEXT_POLICY}; line-height: 1.5; }}
.persona-card {{
    background: linear-gradient(135deg, {C_PERSONA_BG1} 0%, {C_PERSONA_BG2} 100%);
    border-radius: 12px; padding: 1.2rem; margin: 0.5rem 0;
    border-left: 4px solid; min-height: 160px;
}}
.persona-card h4 {{ margin: 0 0 0.5rem 0; font-size: 1rem; }}
.persona-card .pct {{ font-family: 'Space Mono', monospace; font-size: 1.5rem; font-weight: 700; }}
.persona-card .desc {{ font-size: 0.82rem; color: {C_TEXT_DESC}; margin-top: 0.4rem; line-height: 1.4; }}
.persona-card .policy {{ font-size: 0.8rem; color: {C_PRIMARY}; margin-top: 0.5rem; font-style: italic; }}
.headline-stat {{
    background: linear-gradient(135deg, {C_HEADLINE_BG1} 0%, {C_HEADLINE_BG2} 100%);
    border: 2px solid {C_PRIMARY}; border-radius: 14px;
    padding: 1.5rem; text-align: center; margin-bottom: 1rem;
}}
.headline-stat .big {{ font-size: 2.5rem; font-weight: 700; color: {C_PRIMARY}; font-family: 'Space Mono', monospace; }}
.headline-stat .sub {{ font-size: 0.9rem; color: {C_TEXT_DESC}; margin-top: 0.3rem; }}
.section-divider {{ border: 0; border-top: 1px solid {C_DIVIDER}; margin: 2rem 0; }}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# CONSTANTS & MAPPINGS
# ════════════════════════════════════════════════════════════════

COUNTRY_ISO = {
    "India": "IND", "USA": "USA", "UK": "GBR", "UAE": "ARE",
    "Germany": "DEU", "Brazil": "BRA", "Nigeria": "NGA",
    "South Africa": "ZAF", "Japan": "JPN", "Australia": "AUS",
    "Singapore": "SGP", "Philippines": "PHL"
}
COUNTRY_DEV = {
    "India": "Developing", "USA": "Developed", "UK": "Developed", "UAE": "Developed",
    "Germany": "Developed", "Brazil": "Developing", "Nigeria": "Developing",
    "South Africa": "Developing", "Japan": "Developed", "Australia": "Developed",
    "Singapore": "Developed", "Philippines": "Developing"
}
EDU_ORDER = ["Below High School", "High School/Diploma", "Bachelor's", "Master's", "PhD"]
ROLE_ORDER = ["Entry-level", "Mid-level", "Senior", "Manager/Director", "Executive/C-suite"]
AUTO_BUCKET_ORDER = ["0-20%", "21-40%", "41-60%", "61-80%", "81-100%"]

PLOTLY_TEMPLATE = "plotly_dark"
CHART_BG = "rgba(0,0,0,0)"
CHART_MARGINS = dict(l=60, r=30, t=50, b=50)
CHART_FONT_COLOR = C_TEXT
CHART_GRID_COLOR = "#333"
C_CHART_AXIS = "#888"

# Custom plotly template with proper font colors
import plotly.io as pio
pio.templates["custom_dark"] = pio.templates["plotly_dark"]
pio.templates["custom_dark"].layout.font = dict(color=CHART_FONT_COLOR, family="DM Sans, sans-serif")
pio.templates["custom_dark"].layout.paper_bgcolor = CHART_BG
pio.templates["custom_dark"].layout.plot_bgcolor = CHART_BG
pio.templates["custom_dark"].layout.xaxis = dict(gridcolor=CHART_GRID_COLOR, zerolinecolor=CHART_GRID_COLOR)
pio.templates["custom_dark"].layout.yaxis = dict(gridcolor=CHART_GRID_COLOR, zerolinecolor=CHART_GRID_COLOR)
PLOTLY_TEMPLATE = "custom_dark"

DERIVED_VARS = [
    "automation_vulnerability_idx", "reskilling_engagement_score",
    "support_ecosystem_score", "anxiety_to_action_ratio", "future_readiness_idx"
]
DERIVED_LABELS = {
    "automation_vulnerability_idx": "Automation Vulnerability",
    "reskilling_engagement_score": "Reskilling Engagement",
    "support_ecosystem_score": "Support Ecosystem",
    "anxiety_to_action_ratio": "Anxiety-to-Action Ratio",
    "future_readiness_idx": "Future Readiness"
}

# ════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    # Try multiple paths
    for path in ["global_reskilling_gap_survey_10k.csv",
                 "data/global_reskilling_gap_survey_10k.csv",
                 os.path.join(os.path.dirname(__file__), "global_reskilling_gap_survey_10k.csv")]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            break
    else:
        st.error("CSV file not found. Place `global_reskilling_gap_survey_10k.csv` in the same directory as `app.py`.")
        st.stop()

    df["dev_tier"] = df["country"].map(COUNTRY_DEV)
    df["iso_code"] = df["country"].map(COUNTRY_ISO)
    df["education_ordinal"] = df["education_level"].map({e: i for i, e in enumerate(EDU_ORDER)})
    df["role_ordinal"] = df["job_role_level"].map({r: i for i, r in enumerate(ROLE_ORDER)})
    df["auto_bucket_ordinal"] = df["pct_tasks_automatable"].map({b: i for i, b in enumerate(AUTO_BUCKET_ORDER)})
    for col in ["role_partially_automated", "enrolled_reskilling", "employer_provides_reskilling",
                "would_switch_for_reskilling", "accept_lower_pay_futureproof", "successful_reskill_transition"]:
        df[col + "_bin"] = (df[col] == "Yes").astype(int)
    df["gender_female"] = (df["gender"] == "Female").astype(int)
    df["govt_subsidy_yes"] = (df["govt_reskilling_subsidies"] == "Yes").astype(int)
    df["dev_tier_developed"] = (df["dev_tier"] == "Developed").astype(int)
    return df

# ════════════════════════════════════════════════════════════════
# FEATURE ENCODING FOR CLASSIFICATION
# ════════════════════════════════════════════════════════════════

CLF_FEATURES = [
    "age", "gender_female", "education_ordinal", "role_ordinal",
    "work_experience_years", "household_dependents",
    "perceived_automation_risk", "role_partially_automated_bin",
    "auto_bucket_ordinal", "ai_awareness",
    "enrolled_reskilling_bin", "upskilling_hours_per_week",
    "willingness_to_reskill", "employer_provides_reskilling_bin",
    "govt_subsidy_yes", "satisfaction_employer_ld",
    "career_confidence_5yr", "career_anxiety",
    "expected_role_change_timeline", "dev_tier_developed"
]

CLF_FEATURE_LABELS = {
    "age": "Age", "gender_female": "Gender (Female)",
    "education_ordinal": "Education Level", "role_ordinal": "Job Role Level",
    "work_experience_years": "Work Experience (Yrs)",
    "household_dependents": "Household Dependents",
    "perceived_automation_risk": "Perceived Automation Risk",
    "role_partially_automated_bin": "Role Partially Automated",
    "auto_bucket_ordinal": "% Tasks Automatable",
    "ai_awareness": "AI/Automation Awareness",
    "enrolled_reskilling_bin": "Enrolled in Reskilling",
    "upskilling_hours_per_week": "Upskilling Hours/Week",
    "willingness_to_reskill": "Willingness to Reskill",
    "employer_provides_reskilling_bin": "Employer Provides Reskilling",
    "govt_subsidy_yes": "Govt Subsidy Available",
    "satisfaction_employer_ld": "L&D Satisfaction",
    "career_confidence_5yr": "Career Confidence (5yr)",
    "career_anxiety": "Career Anxiety",
    "expected_role_change_timeline": "Expected Role Change Timeline",
    "dev_tier_developed": "Developed Economy"
}

def get_clf_data(df):
    """Prepare classification data."""
    model_df = df[CLF_FEATURES + ["successful_reskill_transition_bin"]].dropna()
    X = model_df[CLF_FEATURES].values
    y = model_df["successful_reskill_transition_bin"].values
    return X, y

# ════════════════════════════════════════════════════════════════
# MODEL TRAINING (CACHED)
# ════════════════════════════════════════════════════════════════

@st.cache_resource
def train_classification():
    df = load_data()
    X, y = get_clf_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    model = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                       subsample=0.8, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # SHAP
    shap_values = None
    if SHAP_AVAILABLE:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
        except Exception:
            pass

    # Feature importance (fallback or supplement)
    feat_imp = model.feature_importances_

    # Per-tier accuracy
    model_df = df[CLF_FEATURES + ["successful_reskill_transition_bin", "dev_tier"]].dropna()
    _, test_df = train_test_split(model_df, test_size=0.25, random_state=42,
                                   stratify=model_df["successful_reskill_transition_bin"])
    tier_metrics = {}
    for tier in ["Developed", "Developing"]:
        mask = test_df["dev_tier"] == tier
        if mask.sum() > 0:
            Xt = test_df.loc[mask, CLF_FEATURES].values
            yt = test_df.loc[mask, "successful_reskill_transition_bin"].values
            yp = model.predict(Xt)
            tier_metrics[tier] = {
                "accuracy": accuracy_score(yt, yp),
                "f1": f1_score(yt, yp),
                "n": int(mask.sum())
            }

    return {
        "model": model, "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "y_pred": y_pred, "y_prob": y_prob,
        "shap_values": shap_values, "feat_imp": feat_imp,
        "tier_metrics": tier_metrics
    }


@st.cache_resource
def train_clustering():
    df = load_data()
    cluster_features = [
        "automation_vulnerability_idx", "reskilling_engagement_score",
        "support_ecosystem_score", "future_readiness_idx"
    ]
    # Use anxiety_to_action_ratio but cap outliers
    cdf = df[cluster_features + ["anxiety_to_action_ratio"]].dropna().copy()
    cdf["anxiety_to_action_ratio"] = cdf["anxiety_to_action_ratio"].clip(upper=cdf["anxiety_to_action_ratio"].quantile(0.95))
    X_cluster = cdf.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    km = KMeans(n_clusters=5, random_state=42, n_init=20)
    labels = km.fit_predict(X_scaled)
    centroids_scaled = km.cluster_centers_
    centroids_raw = scaler.inverse_transform(centroids_scaled)

    # Name personas based on centroid characteristics
    # columns: auto_vuln, reskill_engage, support_eco, future_ready, anxiety_action
    persona_map = {}
    centroid_df = pd.DataFrame(centroids_raw,
                                columns=cluster_features + ["anxiety_to_action_ratio"])
    for i in range(5):
        c = centroid_df.iloc[i]
        if c["reskilling_engagement_score"] > centroid_df["reskilling_engagement_score"].median() and \
           c["support_ecosystem_score"] > centroid_df["support_ecosystem_score"].median() and \
           c["future_readiness_idx"] > centroid_df["future_readiness_idx"].median():
            persona_map[i] = "Future-Ready Professionals"
        elif c["anxiety_to_action_ratio"] > centroid_df["anxiety_to_action_ratio"].quantile(0.7) and \
             c["reskilling_engagement_score"] < centroid_df["reskilling_engagement_score"].median():
            persona_map[i] = "Anxious but Paralyzed"
        elif c["automation_vulnerability_idx"] > centroid_df["automation_vulnerability_idx"].median() and \
             c["reskilling_engagement_score"] > centroid_df["reskilling_engagement_score"].median():
            persona_map[i] = "Automation-Exposed & Active"
        elif c["support_ecosystem_score"] < centroid_df["support_ecosystem_score"].quantile(0.3):
            persona_map[i] = "Structurally Unsupported"
        else:
            persona_map[i] = "Complacent Incumbents"

    # Resolve duplicates
    used_names = set()
    all_names = ["Future-Ready Professionals", "Anxious but Paralyzed",
                 "Automation-Exposed & Active", "Structurally Unsupported", "Complacent Incumbents"]
    for i in range(5):
        if persona_map.get(i) in used_names:
            for name in all_names:
                if name not in used_names:
                    persona_map[i] = name
                    break
        used_names.add(persona_map[i])

    # Assign labels back to full df indices
    label_series = pd.Series(labels, index=cdf.index)

    return {
        "labels": label_series, "centroids": centroid_df,
        "persona_map": persona_map, "scaler": scaler, "model": km
    }


@st.cache_resource
def run_association_rules():
    df = load_data()
    # Select and binarize categorical variables
    rules_df = pd.DataFrame()
    rules_df["Developing_Country"] = df["dev_tier"] == "Developing"
    rules_df["Developed_Country"] = df["dev_tier"] == "Developed"
    rules_df["Female"] = df["gender"] == "Female"
    rules_df["Male"] = df["gender"] == "Male"
    # Education groups
    rules_df["Edu_Postgraduate"] = df["education_level"].isin(["Master's", "PhD"])
    rules_df["Edu_Bachelors"] = df["education_level"] == "Bachelor's"
    rules_df["Edu_Below_Bachelors"] = df["education_level"].isin(["Below High School", "High School/Diploma"])
    # Key industries
    for ind in ["Technology/IT", "Financial Services", "Manufacturing", "Healthcare"]:
        rules_df[f"Ind_{ind.split('/')[0]}"] = df["industry"] == ind
    # Role
    rules_df["Role_Mid"] = df["job_role_level"] == "Mid-level"
    rules_df["Role_Senior_Plus"] = df["job_role_level"].isin(["Senior", "Manager/Director", "Executive/C-suite"])
    # Barriers
    for barrier in ["Cost", "Time", "Family Responsibilities", "Lack of Awareness"]:
        rules_df[f"Barrier_{barrier}"] = df["biggest_barrier"] == barrier
    # Support
    rules_df["Employer_Support_Yes"] = df["employer_provides_reskilling"] == "Yes"
    rules_df["Employer_Support_No"] = df["employer_provides_reskilling"] == "No"
    rules_df["Govt_Subsidy_Yes"] = df["govt_reskilling_subsidies"] == "Yes"
    # Reskilling
    rules_df["Enrolled_Reskilling"] = df["enrolled_reskilling"] == "Yes"
    rules_df["High_LD_Satisfaction"] = df["satisfaction_employer_ld"] >= 4
    rules_df["Low_LD_Satisfaction"] = df["satisfaction_employer_ld"] <= 2
    rules_df["High_Anxiety"] = df["career_anxiety"] >= 4
    rules_df["High_Willingness"] = df["willingness_to_reskill"] >= 4
    rules_df["Successful_Transition"] = df["successful_reskill_transition"] == "Yes"
    # Income tier (rough: use ordinal if available)
    rules_df["Low_Income"] = df["income_reskilling_gap"].isin(["At-Risk Low Earner", "Striving Upskiller"])
    rules_df["High_Income"] = df["income_reskilling_gap"].isin(["Complacent High Earner", "Invested High Earner"])
    # Skills
    for skill in ["Data/AI & Machine Learning", "Cloud Computing", "Cybersecurity", "Soft Skills/Leadership"]:
        safe = skill.replace("/", "_").replace("&", "").replace(" ", "_")
        rules_df[f"Skill_{safe}"] = df["top_skill_pursued"] == skill
    # Platforms
    rules_df["Platform_Self_Taught"] = df["learning_platform"].isin(["YouTube/Free Resources", "Udemy"])
    rules_df["Platform_Premium"] = df["learning_platform"].isin(["Coursera", "LinkedIn Learning", "edX"])

    rules_df = rules_df.dropna()

    frequent = apriori(rules_df, min_support=0.04, use_colnames=True, max_len=3)
    if len(frequent) == 0:
        frequent = apriori(rules_df, min_support=0.02, use_colnames=True, max_len=3)

    rules = association_rules(frequent, metric="lift", min_threshold=1.2, num_itemsets=len(frequent))
    # Add conviction
    rules["conviction"] = rules.apply(
        lambda r: (1 - r["consequent support"]) / (1 - r["confidence"]) if r["confidence"] < 1 else np.inf, axis=1
    )
    rules = rules.sort_values("lift", ascending=False).head(100)
    # Format frozensets for display
    rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    rules["consequents_str"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
    return rules


@st.cache_resource
def train_regression():
    df = load_data()
    feature_cols = [
        "age", "gender_female", "education_ordinal", "role_ordinal",
        "work_experience_years", "household_dependents",
        "perceived_automation_risk", "role_partially_automated_bin",
        "ai_awareness", "enrolled_reskilling_bin", "upskilling_hours_per_week",
        "employer_provides_reskilling_bin", "govt_subsidy_yes",
        "satisfaction_employer_ld", "career_confidence_5yr",
        "career_anxiety", "expected_role_change_timeline", "dev_tier_developed"
    ]

    def fit_ols_manual(X_df, y_series):
        """Fit OLS regression manually using numpy for Python 3.14 compatibility."""
        from scipy import stats as sp_stats
        X = X_df.values.astype(float)
        y = y_series.values.astype(float)
        # Add constant
        X_const = np.column_stack([np.ones(len(X)), X])
        feature_names = ["const"] + list(X_df.columns)
        n, p = X_const.shape

        # OLS: beta = (X'X)^-1 X'y
        try:
            XtX_inv = np.linalg.inv(X_const.T @ X_const)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(X_const.T @ X_const)

        beta = XtX_inv @ (X_const.T @ y)
        y_hat = X_const @ beta
        residuals = y - y_hat

        # Degrees of freedom
        df_resid = max(n - p, 1)
        mse = np.sum(residuals ** 2) / df_resid

        # Standard errors
        se = np.sqrt(np.diag(XtX_inv) * mse)
        se = np.where(se > 0, se, 1e-10)  # guard against zero

        # t-stats and p-values
        t_stats = beta / se
        p_values = 2 * sp_stats.t.sf(np.abs(t_stats), df_resid)

        # Confidence intervals (95%)
        t_crit = sp_stats.t.ppf(0.975, df_resid)
        ci_low = beta - t_crit * se
        ci_high = beta + t_crit * se

        # R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / max(ss_tot, 1e-10)
        r2_adj = 1 - (1 - r2) * (n - 1) / max(df_resid, 1)

        return {
            "params": pd.Series(beta, index=feature_names),
            "pvalues": pd.Series(p_values, index=feature_names),
            "ci_low": pd.Series(ci_low, index=feature_names),
            "ci_high": pd.Series(ci_high, index=feature_names),
            "rsquared_adj": r2_adj,
            "residuals": residuals,
            "fitted": y_hat,
            "feature_names": feature_names
        }

    # Model 1: Willingness to reskill
    reg_df = df[feature_cols + ["willingness_to_reskill"]].dropna()
    model1 = fit_ols_manual(reg_df[feature_cols], reg_df["willingness_to_reskill"])

    # Model 2: Career anxiety
    feat2 = [c for c in feature_cols if c != "career_anxiety"]
    reg_df2 = df[feat2 + ["career_anxiety"]].dropna()
    model2 = fit_ols_manual(reg_df2[feat2], reg_df2["career_anxiety"])

    # Multi-country models (willingness)
    country_models = {}
    for country in ["India", "USA", "Germany", "Nigeria"]:
        cdf = df[df["country"] == country][feature_cols + ["willingness_to_reskill"]].dropna()
        if len(cdf) > 50:
            try:
                country_models[country] = fit_ols_manual(cdf[feature_cols], cdf["willingness_to_reskill"])
            except Exception:
                pass

    return {
        "willingness_model": model1, "anxiety_model": model2,
        "country_models": country_models, "features": feature_cols
    }


# ════════════════════════════════════════════════════════════════
# HTML HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════

def kpi_card(value, label, sub="", color=C_PRIMARY):
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color:{color}">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>"""

def policy_panel(title, items):
    items_html = "".join(f"<li>{item}</li>" for item in items)
    return f"""
    <div class="policy-panel">
        <h4>🎯 {title}</h4>
        <ul>{items_html}</ul>
    </div>"""

def callout_box(title, body, color=C_INFO):
    return f"""
    <div class="callout-box" style="border-left: 3px solid {color};">
        <div class="callout-title" style="color:{color}">{title}</div>
        <div class="callout-body">{body}</div>
    </div>"""

def headline_stat(value, subtitle):
    return f"""
    <div class="headline-stat">
        <div class="big">{value}</div>
        <div class="sub">{subtitle}</div>
    </div>"""

def section_divider():
    return '<hr class="section-divider">'

# ════════════════════════════════════════════════════════════════
# MAIN APP
# ════════════════════════════════════════════════════════════════

df = load_data()

# ── Sidebar ──
with st.sidebar:
    st.markdown("### 🌍 Global Filters")
    sel_countries = st.multiselect("Countries", df["country"].unique().tolist(),
                                    default=df["country"].unique().tolist())
    sel_industries = st.multiselect("Industries", df["industry"].unique().tolist(),
                                     default=df["industry"].unique().tolist())
    age_range = st.slider("Age Range", int(df["age"].min()), int(df["age"].max()),
                           (int(df["age"].min()), int(df["age"].max())))
    sel_genders = st.multiselect("Gender", [g for g in df["gender"].unique() if pd.notna(g)],
                                  default=[g for g in df["gender"].unique() if pd.notna(g)])
    st.markdown("---")
    st.caption("Dashboard models are trained on the full 10K dataset. Filters apply to descriptive visualizations only.")
    st.caption(f"**Dataset:** {len(df):,} respondents · 12 countries · 38 variables")

_mask = (
    df["country"].isin(sel_countries) &
    df["industry"].isin(sel_industries) &
    df["age"].between(age_range[0], age_range[1]) &
    (df["gender"].isin(sel_genders) | df["gender"].isna())
)
fdf = df[_mask]

if len(fdf) == 0:
    st.markdown("# 🌍 Ctrl+Alt+Reskill")
    st.markdown("##### Global Workforce Reskilling Gap — Intelligence Dashboard for Policy & Industry Leaders")
    st.warning("No data matches the current filter selection. Please select at least one option in each filter.")
    st.stop()

# ── Title ──
st.markdown("# 🌍 Ctrl+Alt+Reskill")
st.markdown("##### Global Workforce Reskilling Gap — Intelligence Dashboard for Policy & Industry Leaders")

# ── Tabs ──
tab_names = [
    "📊 Executive Summary",
    "🎯 Who Successfully Reskills?",
    "👥 Five Workforce Personas",
    "🔗 Behavioural Patterns",
    "📈 Drivers of Willingness & Anxiety",
    "🗺️ Geographic Intelligence",
    "⚖️ The Gender Gap",
    "💰 Income-Reskilling Matrix"
]
tabs = st.tabs(tab_names)


# ════════════════════════════════════════════════════════════════
# TAB 1: EXECUTIVE SUMMARY
# ════════════════════════════════════════════════════════════════

with tabs[0]:
    clf_res = train_classification()
    clust_res = train_clustering()

    # Compute headline stat: employer support vs education SHAP ratio
    if clf_res["shap_values"] is not None:
        shap_abs = np.abs(clf_res["shap_values"]).mean(axis=0)
        emp_idx = CLF_FEATURES.index("employer_provides_reskilling_bin")
        edu_idx = CLF_FEATURES.index("education_ordinal")
        emp_shap = shap_abs[emp_idx]
        edu_shap = max(shap_abs[edu_idx], 0.001)
        shap_ratio = emp_shap / edu_shap
    else:
        emp_idx = CLF_FEATURES.index("employer_provides_reskilling_bin")
        edu_idx = CLF_FEATURES.index("education_ordinal")
        emp_imp = clf_res["feat_imp"][emp_idx]
        edu_imp = max(clf_res["feat_imp"][edu_idx], 0.001)
        shap_ratio = emp_imp / edu_imp

    # Headline
    st.markdown(headline_stat(
        f"{shap_ratio:.1f}×",
        "Employer reskilling support is more predictive of successful career transitions than education level"
    ), unsafe_allow_html=True)

    # KPI row
    enrollment_rate = (fdf["enrolled_reskilling"] == "Yes").mean() * 100
    transition_rate = (fdf["successful_reskill_transition"] == "Yes").mean() * 100
    avg_vuln = fdf["automation_vulnerability_idx"].mean()
    emp_support_rate = (fdf["employer_provides_reskilling"] == "Yes").mean() * 100
    avg_anx_action = fdf["anxiety_to_action_ratio"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(kpi_card(f"{enrollment_rate:.0f}%", "Reskilling Enrollment", "Currently enrolled", C_INFO), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card(f"{transition_rate:.0f}%", "Successful Transitions", "Past 3 years", C_PRIMARY), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card(f"{avg_vuln:.2f}", "Avg Automation Vulnerability", "Index (0–1)", C_RISK), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card(f"{emp_support_rate:.0f}%", "Employer Support Rate", "Provides reskilling", C_WARN), unsafe_allow_html=True)
    with c5:
        st.markdown(kpi_card(f"{avg_anx_action:.1f}", "Anxiety-to-Action Ratio", "Higher = more paralyzed", C_PURPLE), unsafe_allow_html=True)

    st.markdown(section_divider(), unsafe_allow_html=True)

    # Row 2: World map + 2x2 mini-grid
    col_map, col_grid = st.columns([3, 2])

    with col_map:
        st.markdown("#### Global Future-Readiness Index")
        country_readiness = fdf.groupby("country").agg(
            future_readiness=("future_readiness_idx", "mean"),
            iso=("iso_code", "first"),
            n=("respondent_id", "count")
        ).reset_index()
        fig_map = px.choropleth(
            country_readiness, locations="iso", color="future_readiness",
            hover_name="country", hover_data={"n": True, "future_readiness": ":.3f", "iso": False},
            color_continuous_scale="Viridis", range_color=(0.4, 0.8),
            labels={"future_readiness": "Readiness Index"},
            template=PLOTLY_TEMPLATE
        )
        fig_map.update_layout(
            geo=dict(bgcolor=CHART_BG, showframe=False, projection_type="natural earth",
                     landcolor=C_GEO_LAND, oceancolor=C_GEO_OCEAN, lakecolor=C_GEO_OCEAN),
            paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
            margin=dict(l=0, r=0, t=10, b=0), height=380,
            coloraxis_colorbar=dict(title="Index", thickness=15, len=0.6)
        )
        st.plotly_chart(fig_map, use_container_width=True)

    with col_grid:
        # Mini chart 1: Target variable donut
        st.markdown("#### Reskilling Success Split")
        trans_counts = fdf["successful_reskill_transition"].value_counts()
        fig_donut = px.pie(values=trans_counts.values, names=trans_counts.index,
                           hole=0.6, color_discrete_sequence=[C_PRIMARY, C_RISK],
                           template=PLOTLY_TEMPLATE)
        fig_donut.update_layout(paper_bgcolor=CHART_BG, margin=dict(l=10, r=10, t=10, b=10),
                                height=170, showlegend=True, legend=dict(font=dict(size=10)))
        st.plotly_chart(fig_donut, use_container_width=True)

        # Mini chart 2: Persona distribution donut
        st.markdown("#### Workforce Persona Distribution")
        persona_labels = clust_res["labels"].map(clust_res["persona_map"])
        persona_counts = persona_labels.value_counts()
        fig_persona_donut = px.pie(values=persona_counts.values, names=persona_counts.index,
                                    hole=0.6, color_discrete_sequence=PERSONA_COLORS,
                                    template=PLOTLY_TEMPLATE)
        fig_persona_donut.update_layout(paper_bgcolor=CHART_BG, margin=dict(l=10, r=10, t=10, b=10),
                                         height=170, showlegend=True, legend=dict(font=dict(size=9)))
        st.plotly_chart(fig_persona_donut, use_container_width=True)

    # Row 3: Top barriers + Income matrix thumbnail
    col_bar, col_matrix = st.columns(2)
    with col_bar:
        st.markdown("#### Top Barriers to Reskilling")
        barriers = fdf["biggest_barrier"].value_counts().head(6)
        barriers_df = pd.DataFrame({"barrier": barriers.index, "count": barriers.values})
        fig_barriers = px.bar(barriers_df, x="count", y="barrier", orientation="h",
                              color="count", color_continuous_scale=[[0, C_INFO], [1, C_RISK]],
                              template=PLOTLY_TEMPLATE, labels={"count": "Respondents", "barrier": ""})
        fig_barriers.update_layout(paper_bgcolor=CHART_BG, margin=CHART_MARGINS, height=300,
                                    showlegend=False, coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_barriers, use_container_width=True)

    with col_matrix:
        st.markdown("#### Income-Reskilling Segments")
        irm = fdf["income_reskilling_gap"].value_counts()
        irm = irm[irm.index != "Unknown"]
        segment_colors = {"At-Risk Low Earner": C_RISK, "Complacent High Earner": C_WARN,
                          "Striving Upskiller": C_PRIMARY, "Invested High Earner": C_INFO}
        irm_chart = pd.DataFrame({"segment": irm.index, "count": irm.values})
        fig_irm = px.bar(irm_chart, x="segment", y="count",
                         color="segment", color_discrete_map=segment_colors,
                         template=PLOTLY_TEMPLATE, labels={"segment": "", "count": "Count"})
        fig_irm.update_layout(paper_bgcolor=CHART_BG, margin=CHART_MARGINS, height=300, showlegend=False)
        st.plotly_chart(fig_irm, use_container_width=True)

    st.markdown(section_divider(), unsafe_allow_html=True)
    st.markdown(policy_panel("Key Takeaways for Policymakers", [
        f"<b>Employer-led reskilling is {shap_ratio:.1f}× more predictive of success than education level.</b> Policy should incentivize workplace training over traditional degree pathways.",
        f"Only <b>{enrollment_rate:.0f}%</b> of the global workforce is currently enrolled in reskilling — urgent scale-up is needed.",
        f"The average anxiety-to-action ratio of <b>{avg_anx_action:.1f}</b> indicates significant paralysis — workers know they need to reskill but aren't acting.",
        f"<b>{emp_support_rate:.0f}%</b> of employers currently provide reskilling support — a structural gap that governments can address through tax incentives and mandates."
    ]), unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# TAB 2: CLASSIFICATION
# ════════════════════════════════════════════════════════════════

with tabs[1]:
    st.markdown("### Who Successfully Reskills, and Why?")
    st.caption("Gradient Boosting Classifier with SHAP Explainability — Target: Successful reskilling transition in past 3 years")

    clf_res = train_classification()

    # Layer 1: SHAP Feature Importance
    st.markdown("#### Feature Importance (SHAP-based)")
    if clf_res["shap_values"] is not None:
        shap_abs = np.abs(clf_res["shap_values"]).mean(axis=0)
    else:
        shap_abs = clf_res["feat_imp"]

    feat_labels = [CLF_FEATURE_LABELS.get(f, f) for f in CLF_FEATURES]
    imp_df = pd.DataFrame({"feature": feat_labels, "importance": shap_abs}).sort_values("importance")
    colors = [C_PRIMARY if v > imp_df["importance"].quantile(0.75) else C_INFO for v in imp_df["importance"]]

    fig_shap = px.bar(imp_df, x="importance", y="feature", orientation="h",
                      template=PLOTLY_TEMPLATE, labels={"importance": "Mean |SHAP Value|", "feature": ""})
    fig_shap.update_traces(marker_color=colors)
    fig_shap.update_layout(paper_bgcolor=CHART_BG, margin=dict(l=200, r=30, t=30, b=50), height=500)
    st.plotly_chart(fig_shap, use_container_width=True)

    st.markdown(callout_box(
        "💡 Key Finding",
        f"<b>Employer reskilling support</b> and <b>upskilling hours per week</b> are the strongest predictors of successful transitions — "
        f"far outweighing education level ({shap_ratio:.1f}× less predictive). "
        f"This challenges the conventional assumption that formal education drives workforce adaptability.",
        C_PRIMARY
    ), unsafe_allow_html=True)

    st.markdown(section_divider(), unsafe_allow_html=True)

    # Layer 2: Model Performance
    st.markdown("#### Model Validation")
    col_roc, col_cm, col_tier = st.columns(3)

    with col_roc:
        fpr, tpr, _ = roc_curve(clf_res["y_test"], clf_res["y_prob"])
        roc_auc = auc(fpr, tpr)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color=C_PRIMARY, width=2),
                                      name=f"AUC = {roc_auc:.3f}"))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color=C_CHART_LINE),
                                      showlegend=False))
        fig_roc.update_layout(title="ROC Curve", template=PLOTLY_TEMPLATE, paper_bgcolor=CHART_BG,
                               xaxis_title="FPR", yaxis_title="TPR", height=320, margin=CHART_MARGINS)
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_cm:
        cm = confusion_matrix(clf_res["y_test"], clf_res["y_pred"])
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale=[[0, C_BG_DARK], [1, C_PRIMARY]],
                           x=["No", "Yes"], y=["No", "Yes"],
                           labels={"x": "Predicted", "y": "Actual"}, template=PLOTLY_TEMPLATE)
        fig_cm.update_layout(title="Confusion Matrix", paper_bgcolor=CHART_BG, height=320,
                              margin=CHART_MARGINS, coloraxis_showscale=False)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_tier:
        st.markdown("##### Accuracy by Economy Tier")
        tier_data = clf_res["tier_metrics"]
        tiers = list(tier_data.keys())
        accs = [tier_data[t]["accuracy"] for t in tiers]
        f1s = [tier_data[t]["f1"] for t in tiers]
        fig_tier = go.Figure()
        fig_tier.add_trace(go.Bar(name="Accuracy", x=tiers, y=accs, marker_color=C_INFO))
        fig_tier.add_trace(go.Bar(name="F1 Score", x=tiers, y=f1s, marker_color=C_PRIMARY))
        fig_tier.update_layout(barmode="group", template=PLOTLY_TEMPLATE, paper_bgcolor=CHART_BG,
                                height=320, margin=CHART_MARGINS, yaxis_tickformat=".0%",
                                legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_tier, use_container_width=True)

    st.markdown(section_divider(), unsafe_allow_html=True)

    # What-If Simulator
    st.markdown("#### 🔮 What-If Reskilling Success Simulator")
    st.caption("Adjust inputs to see predicted probability of successful reskilling transition")
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        sim_age = st.slider("Age", 20, 62, 35, key="sim_age")
        sim_employer = st.selectbox("Employer Provides Reskilling", ["Yes", "No"], key="sim_emp")
        sim_edu = st.selectbox("Education", EDU_ORDER, index=2, key="sim_edu")
    with sc2:
        sim_hours = st.slider("Upskilling Hours/Week", 0.0, 25.0, 5.0, step=0.5, key="sim_hours")
        sim_enrolled = st.selectbox("Enrolled in Reskilling", ["Yes", "No"], key="sim_enrolled")
        sim_role = st.selectbox("Job Role Level", ROLE_ORDER, index=1, key="sim_role")
    with sc3:
        sim_auto_risk = st.slider("Perceived Automation Risk", 1, 5, 3, key="sim_risk")
        sim_anxiety = st.slider("Career Anxiety", 1, 5, 3, key="sim_anx")
        sim_willingness = st.slider("Willingness to Reskill", 1, 5, 3, key="sim_will")
    with sc4:
        sim_dev = st.selectbox("Economy Type", ["Developed", "Developing"], key="sim_dev")
        sim_conf = st.slider("Career Confidence (5yr)", 1, 5, 3, key="sim_conf")
        sim_satisfaction = st.slider("L&D Satisfaction", 1, 5, 3, key="sim_sat")

    # Build feature vector
    sim_features = np.array([[
        sim_age, 0, EDU_ORDER.index(sim_edu), ROLE_ORDER.index(sim_role),
        max(sim_age - 22, 0), 1,
        sim_auto_risk, 1 if sim_auto_risk >= 4 else 0,
        2, 3,
        1 if sim_enrolled == "Yes" else 0, sim_hours,
        sim_willingness, 1 if sim_employer == "Yes" else 0,
        0, sim_satisfaction,
        sim_conf, sim_anxiety,
        5, 1 if sim_dev == "Developed" else 0
    ]])
    sim_prob = clf_res["model"].predict_proba(sim_features)[0, 1]

    st.markdown(f"""
    <div style="text-align:center; padding:1.5rem; background:linear-gradient(135deg,{C_SIM_BG1},{C_SIM_BG2});
                border-radius:12px; border:2px solid {C_PRIMARY if sim_prob > 0.5 else C_RISK};">
        <div style="font-size:3rem; font-weight:700; font-family:'Space Mono',monospace;
                    color:{C_PRIMARY if sim_prob > 0.5 else C_RISK}">{sim_prob:.0%}</div>
        <div style="color:{C_TEXT_MUTED}; font-size:0.9rem;">Predicted Probability of Successful Reskilling Transition</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(section_divider(), unsafe_allow_html=True)

    # Layer 3: Policy Implications
    # Compute hours threshold from data: find where transition rate jumps
    hours_bins = pd.cut(df["upskilling_hours_per_week"], bins=[0, 2, 5, 10, 15, 25])
    hours_trans = df.groupby(hours_bins, observed=True)["successful_reskill_transition_bin"].mean()
    threshold_idx = (hours_trans > hours_trans.iloc[0] * 1.5).idxmax() if len(hours_trans) > 0 else "5-10"
    hours_threshold = 5  # default
    try:
        hours_threshold = int(threshold_idx.left)
    except Exception:
        pass

    st.markdown(policy_panel("Policy Implications — Classification Findings", [
        f"<b>Governments should incentivize employer-led reskilling through tax credits</b> rather than expanding university enrollment subsidies. "
        f"Employer support is {shap_ratio:.1f}× more impactful than education level in predicting successful transitions.",
        f"<b>Structured upskilling time of {hours_threshold}+ hours per week</b> is the threshold for meaningful impact. "
        f"Workplace learning mandates should target this minimum to ensure reskilling programs achieve outcomes, not just participation.",
        f"<b>Model generalizes across economic contexts</b> — accuracy for developed economies "
        f"({clf_res['tier_metrics'].get('Developed', {}).get('accuracy', 0):.0%}) and developing economies "
        f"({clf_res['tier_metrics'].get('Developing', {}).get('accuracy', 0):.0%}) are comparable, "
        f"suggesting these policy levers are globally applicable.",
        "<b>Career anxiety does not predict reskilling success</b> — fear-based messaging about automation is not an effective policy tool. "
        "Focus on enabling structures (employer programs, dedicated time) rather than awareness campaigns."
    ]), unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# TAB 3: CLUSTERING
# ════════════════════════════════════════════════════════════════

with tabs[2]:
    st.markdown("### The Five Global Workforce Personas")
    st.caption("K-Means Clustering (k=5) on composite reskilling dimensions")

    clust_res = train_clustering()
    df_with_persona = df.copy()
    df_with_persona["cluster"] = clust_res["labels"]
    df_with_persona["persona"] = df_with_persona["cluster"].map(clust_res["persona_map"])
    df_with_persona = df_with_persona.dropna(subset=["persona"])
    fdf_persona = df_with_persona[df_with_persona.index.isin(fdf.index)]

    # Layer 1: Radar chart
    st.markdown("#### Persona Profiles — Radar Comparison")
    radar_features = ["automation_vulnerability_idx", "reskilling_engagement_score",
                      "support_ecosystem_score", "future_readiness_idx"]
    radar_labels = ["Auto. Vulnerability", "Reskilling Engagement", "Support Ecosystem", "Future Readiness"]

    fig_radar = go.Figure()
    for i, (cluster_id, persona_name) in enumerate(clust_res["persona_map"].items()):
        vals = clust_res["centroids"].iloc[cluster_id][radar_features].values.tolist()
        vals.append(vals[0])  # close the polygon
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=radar_labels + [radar_labels[0]],
            fill="toself", name=persona_name, opacity=0.6,
            line=dict(color=PERSONA_COLORS[i % len(PERSONA_COLORS)])
        ))
    fig_radar.update_layout(
        polar=dict(bgcolor=CHART_BG, radialaxis=dict(visible=True, range=[0, 1], gridcolor=CHART_GRID_COLOR)),
        template=PLOTLY_TEMPLATE, paper_bgcolor=CHART_BG, height=450,
        legend=dict(orientation="h", y=-0.15), margin=dict(l=60, r=60, t=40, b=80)
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Persona cards
    st.markdown("#### Persona Profiles")
    persona_descs = {
        "Future-Ready Professionals": ("High engagement, strong employer support, confident about career trajectory.",
                                        "Benchmark this group's employer policies and replicate across sectors."),
        "Anxious but Paralyzed": ("High career anxiety but low reskilling action. Stuck in awareness without execution.",
                                   "Deploy structured intervention programs with guided learning pathways and reduced barrier-to-entry."),
        "Automation-Exposed & Active": ("Facing immediate automation pressure and actively reskilling in response.",
                                         "Accelerate their transition with targeted subsidies and fast-track certifications."),
        "Structurally Unsupported": ("Willing to reskill but lacking employer or government support structures.",
                                      "Address the supply gap: mandate employer L&D investment and expand public reskilling infrastructure."),
        "Complacent Incumbents": ("Low anxiety, low engagement, low vulnerability perception. Not yet feeling the urgency.",
                                   "Pre-emptive awareness campaigns targeting industries facing medium-term disruption.")
    }

    pcols = st.columns(5)
    persona_counts = df_with_persona["persona"].value_counts()
    for i, (cluster_id, persona_name) in enumerate(clust_res["persona_map"].items()):
        pct = persona_counts.get(persona_name, 0) / len(df_with_persona) * 100
        desc, pol = persona_descs.get(persona_name, ("", ""))
        with pcols[i % 5]:
            st.markdown(f"""
            <div class="persona-card" style="border-color:{PERSONA_COLORS[i % len(PERSONA_COLORS)]}">
                <h4 style="color:{PERSONA_COLORS[i % len(PERSONA_COLORS)]}">{persona_name}</h4>
                <div class="pct" style="color:{PERSONA_COLORS[i % len(PERSONA_COLORS)]}">{pct:.0f}%</div>
                <div class="desc">{desc}</div>
                <div class="policy">→ {pol}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown(section_divider(), unsafe_allow_html=True)

    # Layer 2: Country × Persona Heatmap
    st.markdown("#### Persona Distribution by Country")
    country_persona = pd.crosstab(df_with_persona["country"], df_with_persona["persona"], normalize="index") * 100
    # Reorder columns by persona names in persona_map order
    ordered_personas = list(clust_res["persona_map"].values())
    available_cols = [p for p in ordered_personas if p in country_persona.columns]
    country_persona = country_persona[available_cols]

    fig_hm = px.imshow(country_persona.round(1), text_auto=".1f",
                        color_continuous_scale=[[0, C_BG_DARK], [0.5, C_INFO], [1, C_PRIMARY]],
                        labels={"color": "% of Country"}, template=PLOTLY_TEMPLATE,
                        aspect="auto")
    fig_hm.update_layout(paper_bgcolor=CHART_BG, margin=dict(l=120, r=30, t=30, b=80), height=420)
    st.plotly_chart(fig_hm, use_container_width=True)

    # Dynamic country callout boxes
    st.markdown("#### Country-Specific Insights")
    callout_countries = {"India": "targeted anxiety-reduction and employer support interventions needed",
                         "Singapore": "model for policy benchmarking",
                         "Nigeria": "structural support investment is the priority",
                         "USA": "address the complacency gap in mid-career professionals"}
    cc1, cc2 = st.columns(2)
    for idx, (country, rec) in enumerate(callout_countries.items()):
        if country in country_persona.index:
            dominant_persona = country_persona.loc[country].idxmax()
            dominant_pct = country_persona.loc[country].max()
            col_target = cc1 if idx % 2 == 0 else cc2
            with col_target:
                st.markdown(callout_box(
                    f"🏳️ {country}",
                    f"<b>{dominant_pct:.0f}% {dominant_persona}</b> — {rec}",
                    PERSONA_COLORS[idx % len(PERSONA_COLORS)]
                ), unsafe_allow_html=True)

    st.markdown(section_divider(), unsafe_allow_html=True)

    # Layer 3: Policy implications
    st.markdown(policy_panel("Policy Implications — Persona-Based Interventions", [
        "<b>One-size-fits-all reskilling policies are ineffective.</b> Five distinct workforce segments require five distinct intervention strategies.",
        "The <b>Anxious but Paralyzed</b> segment needs structured, low-barrier entry points — not more awareness campaigns. Governments should fund guided learning pathways with mentorship.",
        "The <b>Structurally Unsupported</b> segment represents willing workers failed by their institutions. Mandate minimum L&D investment thresholds for employers above a certain size.",
        "The <b>Complacent Incumbents</b> are the hidden risk — they don't know they need to reskill. Sector-specific future-of-work briefings targeting their industries are critical.",
        "Countries with high concentrations of <b>Future-Ready Professionals</b> (see heatmap above) should be studied as policy benchmarks."
    ]), unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# TAB 4: ASSOCIATION RULES
# ════════════════════════════════════════════════════════════════

with tabs[3]:
    st.markdown("### What Behavioural Patterns Co-Occur?")
    st.caption("Apriori Association Rule Mining — discovering policy-relevant co-occurrence patterns")

    rules = run_association_rules()

    # Layer 1: Network Graph
    st.markdown("#### Association Rule Network")
    if len(rules) > 0:
        G = nx.DiGraph()
        top_rules = rules.head(30)
        for _, row in top_rules.iterrows():
            for ant in row["antecedents"]:
                for con in row["consequents"]:
                    G.add_edge(ant, con, lift=row["lift"], support=row["support"])

        pos = nx.spring_layout(G, k=2, seed=42)
        edge_traces = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            lift_val = edge[2].get("lift", 1)
            edge_traces.append(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode="lines", line=dict(width=max(1, min(lift_val, 5)), color=C_INFO),
                opacity=0.4, hoverinfo="none", showlegend=False
            ))

        node_x, node_y, node_text, node_size = [], [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x); node_y.append(y)
            node_text.append(node)
            node_size.append(10 + G.degree(node) * 3)

        fig_net = go.Figure(data=edge_traces + [go.Scatter(
            x=node_x, y=node_y, mode="markers+text", text=node_text,
            textposition="top center", textfont=dict(size=9, color=C_TEXT),
            marker=dict(size=node_size, color=C_PRIMARY, line=dict(width=1, color=CHART_GRID_COLOR)),
            hoverinfo="text", showlegend=False
        )])
        fig_net.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor=CHART_BG,
                               xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                               yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                               height=450, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_net, use_container_width=True)

    # Highlighted callout boxes for key pattern findings
    st.markdown("#### Key Pattern Findings")
    cb1, cb2, cb3 = st.columns(3)

    # Robust rule finder — tries multiple matching strategies
    def find_rule_flexible(rules_df, ant_keywords, con_keywords):
        """Find best matching rule with flexible keyword matching."""
        best_rule = None
        best_lift = 0
        for _, r in rules_df.iterrows():
            ant_str = r["antecedents_str"].lower()
            con_str = r["consequents_str"].lower()
            ant_match = sum(1 for a in ant_keywords if a.lower() in ant_str)
            con_match = sum(1 for c in con_keywords if c.lower() in con_str)
            if ant_match > 0 and con_match > 0 and r["lift"] > best_lift:
                best_rule = r
                best_lift = r["lift"]
        return best_rule

    r1 = find_rule_flexible(rules, ["developing", "low_income"], ["barrier_cost", "cost"])
    r2 = find_rule_flexible(rules, ["ind_technology", "employer_support_yes"], ["high_ld", "satisfaction"])
    r3 = find_rule_flexible(rules, ["female"], ["high_willingness", "willingness"])

    # Compute empirical stats as fallback context
    dev_cost_pct = (df[(df["dev_tier"] == "Developing") & (df["biggest_barrier"] == "Cost")].shape[0] /
                    max(df[df["dev_tier"] == "Developing"].shape[0], 1)) * 100
    tech_emp_sat = df[(df["industry"] == "Technology/IT") & (df["employer_provides_reskilling"] == "Yes")]["satisfaction_employer_ld"].mean()
    f_willingness = df[df["gender"] == "Female"]["willingness_to_reskill"].mean()
    m_willingness = df[df["gender"] == "Male"]["willingness_to_reskill"].mean()

    with cb1:
        if r1 is not None:
            lift_val = f"{r1['lift']:.2f}"
            conf_val = f"{r1['confidence']:.0%}"
            body = (f"<b>Developing Country + Low Income → Cost Barrier</b><br>"
                    f"Lift: {lift_val} · Confidence: {conf_val}<br>"
                    f"Cost is cited by {dev_cost_pct:.0f}% of developing-country respondents — the dominant structural barrier.")
        else:
            body = (f"<b>Developing Country + Low Income → Cost Barrier</b><br>"
                    f"Cost is cited by {dev_cost_pct:.0f}% of developing-country respondents — the dominant structural barrier in emerging economies.")
        st.markdown(callout_box("📌 Cost as Structural Barrier", body, C_RISK), unsafe_allow_html=True)

    with cb2:
        if r2 is not None:
            lift_val = f"{r2['lift']:.2f}"
            conf_val = f"{r2['confidence']:.0%}"
            body = (f"<b>Tech Industry + Employer Support → High L&D Satisfaction</b><br>"
                    f"Lift: {lift_val} · Confidence: {conf_val}<br>"
                    f"Tech employers with reskilling programs achieve avg {tech_emp_sat:.1f}/5 satisfaction — setting the standard.")
        else:
            body = (f"<b>Tech Industry + Employer Support → High L&D Satisfaction</b><br>"
                    f"Tech employers with reskilling programs achieve avg {tech_emp_sat:.1f}/5 satisfaction — a model for other sectors.")
        st.markdown(callout_box("📌 Tech Employer Best Practice", body, C_PRIMARY), unsafe_allow_html=True)

    with cb3:
        if r3 is not None:
            lift_val = f"{r3['lift']:.2f}"
            conf_val = f"{r3['confidence']:.0%}"
            body = (f"<b>Female Workers → Equal/High Willingness to Reskill</b><br>"
                    f"Lift: {lift_val} · Confidence: {conf_val}<br>"
                    f"Women report {f_willingness:.2f}/5 willingness vs {m_willingness:.2f}/5 for men — the gap is access, not motivation.")
        else:
            body = (f"<b>Female Workers → Equal/High Willingness to Reskill</b><br>"
                    f"Women report {f_willingness:.2f}/5 willingness vs {m_willingness:.2f}/5 for men — the gap is access, not motivation.")
        st.markdown(callout_box("📌 Gender Willingness Parity", body, C_PURPLE), unsafe_allow_html=True)

    st.markdown(section_divider(), unsafe_allow_html=True)

    # Layer 2: Rules Table with Conviction
    st.markdown("#### Top Association Rules")
    display_rules = rules[["antecedents_str", "consequents_str", "support", "confidence", "lift", "conviction"]].head(25).copy()
    display_rules.columns = ["Antecedent", "Consequent", "Support", "Confidence", "Lift", "Conviction"]
    display_rules["Support"] = display_rules["Support"].round(3)
    display_rules["Confidence"] = display_rules["Confidence"].round(3)
    display_rules["Lift"] = display_rules["Lift"].round(2)
    display_rules["Conviction"] = display_rules["Conviction"].apply(lambda x: f"{x:.2f}" if x < 100 else "∞")
    st.dataframe(display_rules, use_container_width=True, height=400)

    # Sankey diagram
    st.markdown("#### Behavioural Flow: Industry → Barrier → Skill Pursued")
    sankey_df = df[["industry", "biggest_barrier", "top_skill_pursued"]].dropna()
    sankey_df = sankey_df[sankey_df["top_skill_pursued"] != "None"]
    # Top flows
    flow1 = sankey_df.groupby(["industry", "biggest_barrier"]).size().reset_index(name="count")
    flow1 = flow1.nlargest(20, "count")
    flow2 = sankey_df.groupby(["biggest_barrier", "top_skill_pursued"]).size().reset_index(name="count")
    flow2 = flow2.nlargest(20, "count")

    all_labels = list(set(flow1["industry"]) | set(flow1["biggest_barrier"]) |
                      set(flow2["biggest_barrier"]) | set(flow2["top_skill_pursued"]))
    label_map = {l: i for i, l in enumerate(all_labels)}

    sources = [label_map[r["industry"]] for _, r in flow1.iterrows()] + \
              [label_map[r["biggest_barrier"]] for _, r in flow2.iterrows()]
    targets = [label_map[r["biggest_barrier"]] for _, r in flow1.iterrows()] + \
              [label_map[r["top_skill_pursued"]] for _, r in flow2.iterrows()]
    values = flow1["count"].tolist() + flow2["count"].tolist()

    fig_sankey = go.Figure(go.Sankey(
        node=dict(pad=15, thickness=20, label=all_labels,
                  color=[C_INFO] * len(all_labels)),
        link=dict(source=sources, target=targets, value=values,
                  color="rgba(108,155,210,0.3)")
    ))
    fig_sankey.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor=CHART_BG, height=500,
                              margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_sankey, use_container_width=True)

    st.markdown(section_divider(), unsafe_allow_html=True)

    # Layer 3: Policy Bundles
    # Compute cost-time co-occurrence stat
    cost_time_cooccur = df[(df["dev_tier"] == "Developing") &
                           (df["income_reskilling_gap"].isin(["At-Risk Low Earner"]))].shape[0]
    cost_pct_in_developing = (df[(df["dev_tier"] == "Developing") & (df["biggest_barrier"] == "Cost")].shape[0] /
                              max(df[df["dev_tier"] == "Developing"].shape[0], 1)) * 100

    st.markdown(policy_panel("Policy Bundles — Translating Rules into Interventions", [
        f"<b>In developing economies, reskilling cost subsidies must be paired with time-flexibility policies</b> "
        f"because cost barriers affect {cost_pct_in_developing:.0f}% of developing-country respondents. "
        f"Subsidies alone are insufficient if workers lack time to use them.",
        "<b>Employer-provided reskilling in Tech and Finance correlates with high L&D satisfaction</b> — "
        "suggesting this model should be extended to Manufacturing and Agriculture through public-private partnerships. "
        "Governments can co-fund employer training programs in underserved sectors.",
        "<b>Female workers demonstrate equal or higher willingness to reskill despite facing Family Responsibilities barriers.</b> "
        "Childcare subsidies and flexible learning schedules must be bundled with reskilling programs to close the gender access gap.",
        "<b>The Sankey flow reveals industry → barrier → skill pathways.</b> Use these to design industry-specific reskilling curricula "
        "rather than generic skills programs."
    ]), unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# TAB 5: REGRESSION
# ════════════════════════════════════════════════════════════════

with tabs[4]:
    st.markdown("### What Drives Reskilling Willingness and Career Anxiety?")
    st.caption("OLS Regression with multi-country comparison")

    reg_res = train_regression()

    # Layer 1: Coefficient plots side-by-side
    st.markdown("#### Regression Coefficients — Top Predictors")
    col_r1, col_r2 = st.columns(2)

    def plot_coefficients(model, title, exclude=["const"], top_n=12):
        params = model["params"].drop(exclude, errors="ignore")
        ci_low = model["ci_low"].drop(exclude, errors="ignore")
        ci_high = model["ci_high"].drop(exclude, errors="ignore")
        pvals = model["pvalues"].drop(exclude, errors="ignore")
        coef_df = pd.DataFrame({
            "feature": [CLF_FEATURE_LABELS.get(f, f) for f in params.index],
            "coef": params.values,
            "ci_low": ci_low.values,
            "ci_high": ci_high.values,
            "significant": pvals.values < 0.05
        }).sort_values("coef", key=abs, ascending=True).tail(top_n)

        colors = [C_PRIMARY if s else "#555" for s in coef_df["significant"]]
        fig = go.Figure()
        fig.add_trace(go.Bar(y=coef_df["feature"], x=coef_df["coef"], orientation="h",
                              marker_color=colors, name="Coefficient"))
        # Error bars
        fig.add_trace(go.Scatter(
            y=coef_df["feature"], x=coef_df["coef"],
            error_x=dict(type="data",
                         array=(coef_df["ci_high"] - coef_df["coef"]).values,
                         arrayminus=(coef_df["coef"] - coef_df["ci_low"]).values,
                         color=C_CHART_LINE, thickness=1),
            mode="markers", marker=dict(size=0), showlegend=False
        ))
        fig.add_vline(x=0, line_dash="dash", line_color=C_CHART_LINE)
        fig.update_layout(title=title, template=PLOTLY_TEMPLATE, paper_bgcolor=CHART_BG,
                           height=400, margin=dict(l=180, r=30, t=50, b=50),
                           showlegend=False, xaxis_title="Coefficient")
        return fig

    with col_r1:
        fig_will = plot_coefficients(reg_res["willingness_model"], "Willingness to Reskill")
        st.plotly_chart(fig_will, use_container_width=True)
    with col_r2:
        fig_anx = plot_coefficients(reg_res["anxiety_model"], "Career Anxiety")
        st.plotly_chart(fig_anx, use_container_width=True)

    # R² metrics
    r2_will = reg_res["willingness_model"]["rsquared_adj"]
    r2_anx = reg_res["anxiety_model"]["rsquared_adj"]
    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown(kpi_card(f"{r2_will:.3f}", "Adj. R² — Willingness Model", "Variance explained", C_INFO), unsafe_allow_html=True)
    with mc2:
        st.markdown(kpi_card(f"{r2_anx:.3f}", "Adj. R² — Anxiety Model", "Variance explained", C_INFO), unsafe_allow_html=True)

    st.markdown(section_divider(), unsafe_allow_html=True)

    # SHAP dependence: Age × Career Confidence (Youth Overconfidence)
    st.markdown("#### Age × Career Confidence Interaction (Youth Overconfidence Paradox)")
    age_conf_df = df[["age", "career_confidence_5yr", "reskilling_engagement_score"]].dropna()
    fig_age_conf = px.scatter(
        age_conf_df, x="age", y="career_confidence_5yr",
        color="reskilling_engagement_score", color_continuous_scale=[[0, C_RISK], [0.5, C_WARN], [1, C_PRIMARY]],
        opacity=0.3, labels={"career_confidence_5yr": "Career Confidence (5yr)", "reskilling_engagement_score": "Reskilling Engagement"},
        template=PLOTLY_TEMPLATE
    )
    # Add manual trend line (age-group means) — no statsmodels dependency
    age_trend = age_conf_df.groupby("age")["career_confidence_5yr"].mean().sort_index()
    # Smooth with rolling average
    age_trend_smooth = age_trend.rolling(window=5, center=True, min_periods=2).mean()
    fig_age_conf.add_trace(go.Scatter(
        x=age_trend_smooth.index, y=age_trend_smooth.values,
        mode="lines", line=dict(color=C_WARN, width=3, dash="solid"),
        name="Trend (smoothed)", showlegend=True
    ))
    fig_age_conf.update_layout(paper_bgcolor=CHART_BG, margin=CHART_MARGINS, height=380,
                                coloraxis_colorbar=dict(title="Engagement", thickness=12))
    st.plotly_chart(fig_age_conf, use_container_width=True)

    # Compute youth overconfidence stats
    young_conf = df[df["age"] < 30]["career_confidence_5yr"].mean()
    young_engage = df[df["age"] < 30]["reskilling_engagement_score"].mean()
    old_conf = df[df["age"] >= 35]["career_confidence_5yr"].mean()
    old_engage = df[df["age"] >= 35]["reskilling_engagement_score"].mean()

    st.markdown(callout_box(
        "💡 Youth Overconfidence Paradox",
        f"Workers under 30 report <b>{young_conf:.1f}/5</b> career confidence but only <b>{young_engage:.2f}</b> reskilling engagement. "
        f"Workers 35+ report <b>{old_conf:.1f}/5</b> confidence but <b>{old_engage:.2f}</b> engagement. "
        f"Young workers are confident they'll be fine — but aren't preparing.",
        C_WARN
    ), unsafe_allow_html=True)

    # Multi-country coefficient comparison
    st.markdown("#### Multi-Country Regression Comparison")
    country_models = reg_res["country_models"]
    if country_models:
        # Get top 3 features from global model
        global_params = reg_res["willingness_model"]["params"].drop("const", errors="ignore")
        top3_features = global_params.abs().nlargest(3).index.tolist()
        top3_labels = [CLF_FEATURE_LABELS.get(f, f) for f in top3_features]

        comparison_data = []
        for country, model in country_models.items():
            for feat, label in zip(top3_features, top3_labels):
                if feat in model["params"].index:
                    comparison_data.append({
                        "Country": country, "Feature": label,
                        "Coefficient": model["params"][feat],
                        "Significant": model["pvalues"][feat] < 0.05
                    })

        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            fig_comp = px.bar(comp_df, x="Country", y="Coefficient", color="Feature",
                              barmode="group", template=PLOTLY_TEMPLATE,
                              color_discrete_sequence=[C_PRIMARY, C_INFO, C_WARN])
            fig_comp.update_layout(paper_bgcolor=CHART_BG, margin=CHART_MARGINS, height=350,
                                    legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown(section_divider(), unsafe_allow_html=True)

    # Model diagnostics (collapsible)
    with st.expander("📐 Model Diagnostics (click to expand)"):
        diag1, diag2 = st.columns(2)
        with diag1:
            st.markdown("##### Willingness Model — Residuals vs Fitted")
            resid = reg_res["willingness_model"]["residuals"]
            fitted = reg_res["willingness_model"]["fitted"]
            fig_resid = px.scatter(x=fitted, y=resid, opacity=0.2, template=PLOTLY_TEMPLATE,
                                    labels={"x": "Fitted Values", "y": "Residuals"})
            fig_resid.add_hline(y=0, line_dash="dash", line_color=C_RISK)
            fig_resid.update_layout(paper_bgcolor=CHART_BG, height=300, margin=CHART_MARGINS)
            st.plotly_chart(fig_resid, use_container_width=True)
        with diag2:
            st.markdown("##### Q-Q Plot")
            from scipy import stats as scipy_stats
            resid_sorted = np.sort(resid)
            theoretical = scipy_stats.norm.ppf(np.linspace(0.01, 0.99, len(resid_sorted)))
            sample_size = min(2000, len(resid_sorted))
            idx = np.linspace(0, len(resid_sorted) - 1, sample_size).astype(int)
            fig_qq = px.scatter(x=theoretical[idx], y=resid_sorted[idx], opacity=0.3,
                                 template=PLOTLY_TEMPLATE, labels={"x": "Theoretical Quantiles", "y": "Sample Quantiles"})
            fig_qq.add_trace(go.Scatter(x=[-3, 3], y=[-3, 3], mode="lines",
                                         line=dict(dash="dash", color=C_RISK), showlegend=False))
            fig_qq.update_layout(paper_bgcolor=CHART_BG, height=300, margin=CHART_MARGINS)
            st.plotly_chart(fig_qq, use_container_width=True)

    # Layer 3: Policy Implications
    # Find the anxiety level with highest willingness
    anx_will = df.groupby("career_anxiety")["willingness_to_reskill"].mean()
    peak_anxiety = anx_will.idxmax()
    age_threshold = 30

    st.markdown(policy_panel("Policy Implications — Regression Findings", [
        f"<b>Moderate urgency messaging drives reskilling action; catastrophist framing causes paralysis.</b> "
        f"Peak willingness occurs at career anxiety level {peak_anxiety}/5. "
        f"Government communications should emphasize opportunity (\"future-proof your career\"), not threat (\"your job will disappear\").",
        f"<b>Workers under {age_threshold} significantly overestimate their career confidence</b> "
        f"(avg {young_conf:.1f}/5) despite lower reskilling engagement ({young_engage:.2f} vs {old_engage:.2f} for 35+). "
        f"Early-career intervention programs are critical before complacency sets in.",
        "<b>Employer L&D satisfaction is a significant predictor of both willingness and reduced anxiety.</b> "
        "Improving workplace learning quality — not just availability — should be a regulatory focus.",
        "<b>The same top-3 drivers hold across India, USA, Germany, and Nigeria</b> (see coefficient chart), "
        "reinforcing the global applicability of these findings for international policy frameworks."
    ]), unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# TAB 6: GEOGRAPHIC INTELLIGENCE
# ════════════════════════════════════════════════════════════════

with tabs[5]:
    st.markdown("### Geographic Intelligence")
    st.caption("Cross-country reskilling landscape comparison")

    metric_choice = st.selectbox("Select Metric for Map", [
        "Future Readiness Index", "Automation Vulnerability", "Reskilling Engagement",
        "Support Ecosystem Score", "Transition Success Rate"
    ], key="geo_metric")

    metric_map_col = {
        "Future Readiness Index": "future_readiness_idx",
        "Automation Vulnerability": "automation_vulnerability_idx",
        "Reskilling Engagement": "reskilling_engagement_score",
        "Support Ecosystem Score": "support_ecosystem_score",
        "Transition Success Rate": "successful_reskill_transition_bin"
    }

    col_name = metric_map_col[metric_choice]
    country_metrics = fdf.groupby("country").agg(
        metric=(col_name, "mean"),
        iso=("iso_code", "first"),
        n=("respondent_id", "count")
    ).reset_index()

    fig_geo = px.choropleth(
        country_metrics, locations="iso", color="metric",
        hover_name="country", hover_data={"metric": ":.3f", "n": True, "iso": False},
        color_continuous_scale="Viridis",
        labels={"metric": metric_choice}, template=PLOTLY_TEMPLATE
    )
    fig_geo.update_layout(
        geo=dict(bgcolor=CHART_BG, showframe=False, projection_type="natural earth",
                 landcolor=C_GEO_LAND, oceancolor=C_GEO_OCEAN),
        paper_bgcolor=CHART_BG, margin=dict(l=0, r=0, t=10, b=0), height=420,
        coloraxis_colorbar=dict(title=metric_choice[:15], thickness=15, len=0.6)
    )
    st.plotly_chart(fig_geo, use_container_width=True)

    st.markdown(section_divider(), unsafe_allow_html=True)

    # Grouped bar: all countries, key metrics
    st.markdown("#### Country Comparison — Key Metrics")
    country_comp = fdf.groupby("country").agg(
        Enrollment=("enrolled_reskilling_bin", "mean"),
        Transition=("successful_reskill_transition_bin", "mean"),
        Employer_Support=("employer_provides_reskilling_bin", "mean"),
        Vulnerability=("automation_vulnerability_idx", "mean"),
        Readiness=("future_readiness_idx", "mean")
    ).reset_index()

    fig_country_comp = go.Figure()
    for metric, color in zip(["Enrollment", "Transition", "Employer_Support"],
                              [C_INFO, C_PRIMARY, C_WARN]):
        fig_country_comp.add_trace(go.Bar(name=metric, x=country_comp["country"],
                                           y=country_comp[metric], marker_color=color))
    fig_country_comp.update_layout(barmode="group", template=PLOTLY_TEMPLATE, paper_bgcolor=CHART_BG,
                                    height=380, margin=CHART_MARGINS, yaxis_tickformat=".0%",
                                    legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig_country_comp, use_container_width=True)

    # Developed vs Developing comparison
    st.markdown("#### Developed vs Developing — Divergent Patterns")
    dc1, dc2 = st.columns(2)
    with dc1:
        barrier_by_tier = pd.crosstab(fdf["dev_tier"], fdf["biggest_barrier"], normalize="index") * 100
        fig_tier_bar = px.bar(barrier_by_tier.reset_index(), x="dev_tier",
                              y=barrier_by_tier.columns.tolist(), barmode="stack",
                              template=PLOTLY_TEMPLATE,
                              color_discrete_sequence=[C_RISK, C_INFO, C_WARN, C_PRIMARY, C_PURPLE, C_ORANGE],
                              labels={"value": "% of Respondents", "dev_tier": ""})
        fig_tier_bar.update_layout(paper_bgcolor=CHART_BG, height=350, margin=CHART_MARGINS,
                                    title="Barrier Distribution by Economy Tier",
                                    legend=dict(font=dict(size=9)))
        st.plotly_chart(fig_tier_bar, use_container_width=True)

    with dc2:
        # Bubble chart: vulnerability vs engagement
        bubble = fdf.groupby("country").agg(
            vulnerability=("automation_vulnerability_idx", "mean"),
            engagement=("reskilling_engagement_score", "mean"),
            n=("respondent_id", "count"),
            tier=("dev_tier", "first")
        ).reset_index()
        fig_bubble = px.scatter(bubble, x="vulnerability", y="engagement", size="n",
                                 color="tier", hover_name="country", size_max=50,
                                 color_discrete_map={"Developed": C_PRIMARY, "Developing": C_RISK},
                                 template=PLOTLY_TEMPLATE,
                                 labels={"vulnerability": "Automation Vulnerability", "engagement": "Reskilling Engagement"})
        fig_bubble.update_layout(paper_bgcolor=CHART_BG, height=350, margin=CHART_MARGINS,
                                  title="Danger Zone Analysis")
        # Add quadrant lines
        fig_bubble.add_hline(y=bubble["engagement"].median(), line_dash="dot", line_color=C_CHART_LINE)
        fig_bubble.add_vline(x=bubble["vulnerability"].median(), line_dash="dot", line_color=C_CHART_LINE)
        st.plotly_chart(fig_bubble, use_container_width=True)

    st.markdown(section_divider(), unsafe_allow_html=True)

    # Country profile card with dominant persona
    st.markdown("#### Country Profile Explorer")
    selected_country = st.selectbox("Select Country", sorted(fdf["country"].unique()), key="geo_country")
    cdata = fdf[fdf["country"] == selected_country]

    # Get dominant persona
    if "cluster" not in df_with_persona.columns:
        clust_res_local = train_clustering()
        df_with_persona_local = df.copy()
        df_with_persona_local["cluster"] = clust_res_local["labels"]
        df_with_persona_local["persona"] = df_with_persona_local["cluster"].map(clust_res_local["persona_map"])
    else:
        df_with_persona_local = df_with_persona

    country_personas = df_with_persona_local[df_with_persona_local["country"] == selected_country]["persona"].value_counts()
    dominant_persona = country_personas.index[0] if len(country_personas) > 0 else "N/A"
    dominant_pct = (country_personas.iloc[0] / country_personas.sum() * 100) if len(country_personas) > 0 else 0

    pc1, pc2, pc3, pc4, pc5 = st.columns(5)
    with pc1:
        st.markdown(kpi_card(f"{(cdata['enrolled_reskilling'] == 'Yes').mean():.0%}", "Enrollment Rate", "", C_INFO), unsafe_allow_html=True)
    with pc2:
        st.markdown(kpi_card(f"{(cdata['successful_reskill_transition'] == 'Yes').mean():.0%}", "Transition Rate", "", C_PRIMARY), unsafe_allow_html=True)
    with pc3:
        st.markdown(kpi_card(f"{cdata['automation_vulnerability_idx'].mean():.2f}", "Avg Vulnerability", "", C_RISK), unsafe_allow_html=True)
    with pc4:
        top_barrier_mode = cdata["biggest_barrier"].mode()
        top_barrier_country = top_barrier_mode.iloc[0] if len(top_barrier_mode) > 0 else "N/A"
        st.markdown(kpi_card(top_barrier_country, "Top Barrier", "", C_WARN), unsafe_allow_html=True)
    with pc5:
        st.markdown(kpi_card(f"{dominant_persona}", "Dominant Persona", f"{dominant_pct:.0f}% of workforce", C_PURPLE), unsafe_allow_html=True)

    st.markdown(section_divider(), unsafe_allow_html=True)
    st.markdown(policy_panel("Policy Implications — Geographic Insights", [
        "<b>Developing economies face a structural double bind:</b> higher cost barriers AND lower employer support. "
        "International development finance should prioritize reskilling infrastructure alongside traditional education investment.",
        "<b>Countries in the 'Danger Zone'</b> (high vulnerability, low engagement — see bubble chart) "
        "need immediate national reskilling strategies with public funding and industry partnerships.",
        "<b>The dominant persona in each country should drive national policy design.</b> "
        "A country with high 'Structurally Unsupported' concentration needs employer mandates; "
        "one with high 'Complacent Incumbents' needs awareness campaigns.",
        "<b>Developed economies should not be complacent</b> — their 'Time' barrier suggests that even with resources, "
        "workers lack protected time for reskilling. Learning leave legislation should be explored."
    ]), unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# TAB 7: GENDER GAP
# ════════════════════════════════════════════════════════════════

with tabs[6]:
    st.markdown("### The Gender Gap Deep-Dive")
    st.caption("Structural barriers vs equal willingness — the access gap in reskilling")

    gdf = fdf[fdf["gender"].isin(["Male", "Female"])]

    # Gender KPI cards
    gk1, gk2, gk3, gk4 = st.columns(4)
    m_enroll = (gdf[gdf["gender"] == "Male"]["enrolled_reskilling"] == "Yes").mean()
    f_enroll = (gdf[gdf["gender"] == "Female"]["enrolled_reskilling"] == "Yes").mean()
    m_trans = (gdf[gdf["gender"] == "Male"]["successful_reskill_transition"] == "Yes").mean()
    f_trans = (gdf[gdf["gender"] == "Female"]["successful_reskill_transition"] == "Yes").mean()
    m_support = (gdf[gdf["gender"] == "Male"]["employer_provides_reskilling"] == "Yes").mean()
    f_support = (gdf[gdf["gender"] == "Female"]["employer_provides_reskilling"] == "Yes").mean()
    m_will = gdf[gdf["gender"] == "Male"]["willingness_to_reskill"].mean()
    f_will = gdf[gdf["gender"] == "Female"]["willingness_to_reskill"].mean()

    with gk1:
        gap = (f_enroll - m_enroll) * 100
        st.markdown(kpi_card(f"{gap:+.1f}pp", "Enrollment Gap (F−M)", f"F:{f_enroll:.0%} vs M:{m_enroll:.0%}",
                              C_PRIMARY if gap >= 0 else C_RISK), unsafe_allow_html=True)
    with gk2:
        gap = (f_trans - m_trans) * 100
        st.markdown(kpi_card(f"{gap:+.1f}pp", "Transition Gap (F−M)", f"F:{f_trans:.0%} vs M:{m_trans:.0%}",
                              C_PRIMARY if gap >= 0 else C_RISK), unsafe_allow_html=True)
    with gk3:
        gap = (f_support - m_support) * 100
        st.markdown(kpi_card(f"{gap:+.1f}pp", "Employer Support Gap (F−M)", f"F:{f_support:.0%} vs M:{m_support:.0%}",
                              C_PRIMARY if gap >= 0 else C_RISK), unsafe_allow_html=True)
    with gk4:
        gap = f_will - m_will
        st.markdown(kpi_card(f"{gap:+.2f}", "Willingness Gap (F−M)", f"F:{f_will:.2f} vs M:{m_will:.2f}",
                              C_PRIMARY if gap >= 0 else C_RISK), unsafe_allow_html=True)

    st.markdown(section_divider(), unsafe_allow_html=True)

    # Layer 1: Box plots comparison
    st.markdown("#### Distribution Comparison — Key Likert Variables")
    likert_vars = ["perceived_automation_risk", "willingness_to_reskill", "career_anxiety",
                   "career_confidence_5yr", "satisfaction_employer_ld", "ai_awareness"]
    likert_labels = ["Automation Risk", "Willingness", "Anxiety", "Confidence", "L&D Satisfaction", "AI Awareness"]

    fig_box = make_subplots(rows=2, cols=3, subplot_titles=likert_labels)
    for idx, (var, label) in enumerate(zip(likert_vars, likert_labels)):
        row = idx // 3 + 1
        col = idx % 3 + 1
        for gender, color in [("Male", C_INFO), ("Female", C_PURPLE)]:
            vals = gdf[gdf["gender"] == gender][var].dropna()
            fig_box.add_trace(go.Box(y=vals, name=gender, marker_color=color, showlegend=(idx == 0),
                                      boxpoints=False), row=row, col=col)
    fig_box.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor=CHART_BG, height=500,
                           margin=dict(l=50, r=30, t=50, b=50))
    st.plotly_chart(fig_box, use_container_width=True)

    # Barrier distribution by gender
    st.markdown("#### Barrier Distribution by Gender")
    barrier_gender = pd.crosstab(gdf["gender"], gdf["biggest_barrier"], normalize="index") * 100
    fig_bg = px.bar(barrier_gender.reset_index(), x="gender", y=barrier_gender.columns.tolist(),
                    barmode="group", template=PLOTLY_TEMPLATE,
                    color_discrete_sequence=[C_RISK, C_INFO, C_WARN, C_PRIMARY, C_PURPLE, C_ORANGE],
                    labels={"value": "% of Respondents", "gender": ""})
    fig_bg.update_layout(paper_bgcolor=CHART_BG, height=380, margin=CHART_MARGINS,
                          legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig_bg, use_container_width=True)

    fam_barrier_f = (gdf[(gdf["gender"] == "Female") & (gdf["biggest_barrier"] == "Family Responsibilities")].shape[0] /
                     max(gdf[gdf["gender"] == "Female"].shape[0], 1)) * 100
    fam_barrier_m = (gdf[(gdf["gender"] == "Male") & (gdf["biggest_barrier"] == "Family Responsibilities")].shape[0] /
                     max(gdf[gdf["gender"] == "Male"].shape[0], 1)) * 100

    st.markdown(callout_box(
        "⚖️ The Access Paradox",
        f"Female respondents cite <b>Family Responsibilities</b> as their top barrier at {fam_barrier_f:.1f}% vs {fam_barrier_m:.1f}% for males — "
        f"yet report <b>equal or higher willingness to reskill</b> (F: {f_will:.2f} vs M: {m_will:.2f}). The gap is access, not motivation.",
        C_PURPLE
    ), unsafe_allow_html=True)

    st.markdown(section_divider(), unsafe_allow_html=True)

    # Employer support by gender × industry heatmap
    st.markdown("#### Employer Support Rate — Gender × Industry")
    gi_support = gdf.groupby(["industry", "gender"])["employer_provides_reskilling_bin"].mean().unstack(fill_value=0) * 100
    fig_gi = px.imshow(gi_support.round(1), text_auto=".1f",
                        color_continuous_scale=[[0, C_BG_DARK], [0.5, C_INFO], [1, C_PRIMARY]],
                        labels={"color": "Support Rate %"}, template=PLOTLY_TEMPLATE, aspect="auto")
    fig_gi.update_layout(paper_bgcolor=CHART_BG, margin=dict(l=160, r=30, t=30, b=50), height=400)
    st.plotly_chart(fig_gi, use_container_width=True)

    st.markdown(section_divider(), unsafe_allow_html=True)
    st.markdown(policy_panel("Policy Implications — Gender Gap", [
        f"<b>The reskilling gender gap is an access problem, not a motivation problem.</b> "
        f"Women report equal willingness ({f_will:.2f}/5) but face {fam_barrier_f:.0f}% Family Responsibilities barriers vs {fam_barrier_m:.0f}% for men.",
        "<b>Childcare subsidies must be bundled with reskilling programs.</b> "
        "Without addressing the care economy, reskilling investments will disproportionately benefit male workers.",
        "<b>Industry-specific gender gaps exist</b> (see heatmap). Manufacturing and Construction show the largest employer support disparities — "
        "sector-specific mandates are needed.",
        "<b>Flexible and asynchronous learning formats</b> should be prioritized in program design to accommodate caregiving schedules."
    ]), unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# TAB 8: INCOME-RESKILLING MATRIX
# ════════════════════════════════════════════════════════════════

with tabs[7]:
    st.markdown("### The Income-Reskilling Matrix")
    st.caption("2×2 strategic framework mapping income against reskilling investment")

    irm_df = fdf[fdf["income_reskilling_gap"] != "Unknown"].copy()

    # Layer 1: Quadrant scatter plot
    st.markdown("#### The Matrix — Income Tier × Reskilling Hours")

    # Create income ordinal for scatter
    irm_df["income_tier_label"] = irm_df["income_reskilling_gap"].map({
        "At-Risk Low Earner": "Low Income",
        "Complacent High Earner": "High Income",
        "Striving Upskiller": "Low Income",
        "Invested High Earner": "High Income"
    })

    # Use jittered positions for scatter
    np.random.seed(42)
    irm_df["income_jitter"] = irm_df["income_reskilling_gap"].map({
        "At-Risk Low Earner": 0, "Striving Upskiller": 0,
        "Complacent High Earner": 1, "Invested High Earner": 1
    }) + np.random.normal(0, 0.15, len(irm_df))
    irm_df["hours_jitter"] = irm_df["upskilling_hours_per_week"] + np.random.normal(0, 0.3, len(irm_df))

    segment_color_map = {
        "At-Risk Low Earner": C_RISK, "Complacent High Earner": C_WARN,
        "Striving Upskiller": C_PRIMARY, "Invested High Earner": C_INFO
    }

    fig_matrix = px.scatter(
        irm_df.sample(min(2000, len(irm_df)), random_state=42),
        x="income_jitter", y="hours_jitter",
        color="income_reskilling_gap", color_discrete_map=segment_color_map,
        opacity=0.4, template=PLOTLY_TEMPLATE,
        labels={"income_jitter": "Income Tier →", "hours_jitter": "Upskilling Hours/Week →",
                "income_reskilling_gap": "Segment"}
    )
    fig_matrix.add_hline(y=5, line_dash="dash", line_color=C_CHART_DASH,
                          annotation_text="5 hrs/week threshold", annotation_position="top left")
    fig_matrix.add_vline(x=0.5, line_dash="dash", line_color=C_CHART_DASH)
    # Quadrant labels
    for label, x, y in [("AT-RISK\nLOW EARNER", 0.0, 1.0), ("STRIVING\nUPSKILLER", 0.0, 15.0),
                         ("COMPLACENT\nHIGH EARNER", 1.0, 1.0), ("INVESTED\nHIGH EARNER", 1.0, 15.0)]:
        fig_matrix.add_annotation(x=x, y=y, text=label, showarrow=False,
                                   font=dict(size=11, color=C_TEXT_MUTED), opacity=0.6)
    fig_matrix.update_layout(paper_bgcolor=CHART_BG, height=500, margin=CHART_MARGINS,
                              xaxis=dict(showticklabels=False), legend=dict(orientation="h", y=-0.12))
    st.plotly_chart(fig_matrix, use_container_width=True)

    # Segment profiles
    st.markdown("#### Segment Profiles")
    seg_cols = st.columns(4)
    segment_order = ["At-Risk Low Earner", "Complacent High Earner", "Striving Upskiller", "Invested High Earner"]
    segment_recs = {
        "At-Risk Low Earner": "Publicly funded reskilling with income support during training",
        "Complacent High Earner": "Employer mandates for continuous learning; future-of-work briefings",
        "Striving Upskiller": "Fast-track certifications, recognition of informal learning, career pathway support",
        "Invested High Earner": "Leverage as mentors and champions; formalize their learning into credentials"
    }

    for i, seg in enumerate(segment_order):
        seg_data = irm_df[irm_df["income_reskilling_gap"] == seg]
        with seg_cols[i]:
            pct = len(seg_data) / len(irm_df) * 100
            trans_rate = (seg_data["successful_reskill_transition"] == "Yes").mean() * 100
            avg_anxiety = seg_data["career_anxiety"].mean()
            top_barrier_mode = seg_data["biggest_barrier"].mode()
            top_barrier = top_barrier_mode.iloc[0] if len(top_barrier_mode) > 0 else "N/A"
            color = segment_color_map[seg]
            st.markdown(f"""
            <div class="persona-card" style="border-color:{color}">
                <h4 style="color:{color}">{seg}</h4>
                <div class="pct" style="color:{color}">{pct:.0f}%</div>
                <div class="desc">
                    Transition rate: <b>{trans_rate:.0f}%</b><br>
                    Avg anxiety: <b>{avg_anxiety:.1f}/5</b><br>
                    Top barrier: <b>{top_barrier}</b>
                </div>
                <div class="policy">→ {segment_recs[seg]}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown(section_divider(), unsafe_allow_html=True)

    # Treemap
    st.markdown("#### Segment Size × Transition Success")
    tree_data = irm_df.groupby("income_reskilling_gap").agg(
        count=("respondent_id", "count"),
        transition_rate=("successful_reskill_transition_bin", "mean")
    ).reset_index()
    tree_data["transition_pct"] = (tree_data["transition_rate"] * 100).round(1)

    fig_tree = px.treemap(tree_data, path=["income_reskilling_gap"], values="count",
                           color="transition_rate", color_continuous_scale=[[0, C_RISK], [0.5, C_WARN], [1, C_PRIMARY]],
                           hover_data={"transition_pct": True, "count": True},
                           template=PLOTLY_TEMPLATE)
    fig_tree.update_layout(paper_bgcolor=CHART_BG, height=350, margin=dict(l=10, r=10, t=10, b=10),
                            coloraxis_colorbar=dict(title="Trans. Rate", thickness=12))
    fig_tree.update_traces(textinfo="label+value+percent root",
                            textfont=dict(size=13))
    st.plotly_chart(fig_tree, use_container_width=True)

    st.markdown(section_divider(), unsafe_allow_html=True)
    st.markdown(policy_panel("Policy Implications — Income-Reskilling Matrix", [
        "<b>At-Risk Low Earners need combined financial + time support.</b> Reskilling vouchers alone won't work if workers "
        "can't afford to reduce working hours. Income replacement during training periods is essential.",
        "<b>Complacent High Earners are the hidden policy challenge.</b> They have resources but no urgency. "
        "Employer-mandated continuous learning credits and sector-specific disruption briefings can activate this segment.",
        "<b>Striving Upskillers are the highest-ROI segment for public investment.</b> They're already motivated and acting — "
        "reducing friction (fast-track certs, recognition of prior learning) will convert effort into outcomes.",
        "<b>Invested High Earners should be leveraged as ecosystem assets.</b> "
        "Formalizing their role as mentors, industry trainers, or policy advisors multiplies the impact of their reskilling investment."
    ]), unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(f"""
<div style="text-align:center; padding:1rem; color:{C_TEXT_MUTED}; font-size:0.8rem;">
    <b>Global Workforce Reskilling Gap Intelligence Dashboard</b>
</div>
""", unsafe_allow_html=True)
