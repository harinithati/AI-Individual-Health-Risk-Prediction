from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent
BUNDLE_PATH = BASE / "hwrpa_model_bundle.joblib"

st.set_page_config(
    page_title="HealthAI | HWRPA",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- Visual system -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');
:root { --ink:#eaf2ff; --muted:#9eb0c9; --line:rgba(148,163,184,.18); }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background: radial-gradient(circle at 12% 0%,rgba(20,184,166,.13),transparent 28%), radial-gradient(circle at 92% 5%,rgba(59,130,246,.15),transparent 26%), #08111f; color:var(--ink); }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#101d31,#0b1525); border-right:1px solid var(--line); }
.block-container { padding-top:1.6rem; padding-bottom:2rem; max-width:1440px; }
.hero { padding:30px 34px; border-radius:24px; background:linear-gradient(115deg,#0e4651 0%,#15566b 48%,#283b72 100%); border:1px solid rgba(148,210,230,.25); box-shadow:0 20px 50px rgba(0,0,0,.25); margin-bottom:22px; }
.kicker { color:#9ff4e5; font-size:11px; font-weight:800; letter-spacing:2.3px; text-transform:uppercase; }
.hero h1 { font-family:Manrope,sans-serif; color:white; font-size:clamp(28px,4vw,43px); line-height:1.1; margin:10px 0 9px; font-weight:800; }
.hero p { color:#d7eaf4; font-size:15px; margin:0; max-width:850px; }
.section-title { font-family:Manrope,sans-serif; font-size:22px; font-weight:800; color:#f2f7ff; margin:16px 0 3px; }
.section-sub { color:var(--muted); font-size:13px; margin-bottom:14px; }
.metric-card { background:linear-gradient(145deg,rgba(25,43,66,.96),rgba(14,27,45,.96)); border:1px solid var(--line); border-radius:17px; padding:17px 19px; min-height:108px; }
.metric-label { color:#9eb4ce; font-size:10px; font-weight:800; letter-spacing:1.1px; text-transform:uppercase; }
.metric-value { color:#fff; font-family:Manrope,sans-serif; font-size:27px; font-weight:800; margin-top:7px; }
.metric-foot { color:#8fa6c2; font-size:11px; margin-top:2px; }
.info-box { padding:14px 17px; border-radius:14px; background:rgba(13,148,136,.11); border:1px solid rgba(45,212,191,.28); color:#c9f1eb; margin:10px 0 18px; }
.warning-box { padding:14px 17px; border-radius:14px; background:rgba(180,83,9,.12); border:1px solid rgba(251,191,36,.3); color:#fde7b0; margin:10px 0 18px; }
.small-muted { color:#9eb0c9; font-size:12px; }
div.stButton > button { border:0; border-radius:12px; font-weight:800; color:white; min-height:45px; background:linear-gradient(90deg,#0d9488,#2563eb); }
div.stButton > button:hover { border:1px solid #8be9dc; color:white; }
[data-testid="stTabs"] button { font-weight:700; }
hr { border-color:var(--line); }
</style>
""", unsafe_allow_html=True)

FEATURES = [
    "_AGEG5YR", "SEXVAR", "_BMI5CAT", "EXERANY2", "SMOKE100",
    "DIABETE4", "BPHIGH6", "GENHLTH", "PHYSHLTH", "MENTHLTH", "EDUCA", "INCOME3"
]
LABELS = {
    "_AGEG5YR": "Age group", "SEXVAR": "Sex", "_BMI5CAT": "BMI category",
    "EXERANY2": "Physical activity", "SMOKE100": "100+ cigarettes in lifetime",
    "DIABETE4": "Diabetes status", "BPHIGH6": "High blood pressure",
    "GENHLTH": "General health", "PHYSHLTH": "Physical unhealthy days (0–30)",
    "MENTHLTH": "Mental unhealthy days (0–30)", "EDUCA": "Education level", "INCOME3": "Income category"
}
OPTIONS = {
    "_AGEG5YR": list(range(1, 14)), "SEXVAR": [1, 2], "_BMI5CAT": [1, 2, 3, 4],
    "EXERANY2": [1, 2], "SMOKE100": [1, 2], "DIABETE4": [1, 2, 3, 4],
    "BPHIGH6": [1, 2, 3, 4], "GENHLTH": [1, 2, 3, 4, 5],
    "PHYSHLTH": list(range(0, 31)), "MENTHLTH": list(range(0, 31)),
    "EDUCA": list(range(1, 7)), "INCOME3": list(range(1, 12))
}
HELP = {
    "_AGEG5YR": "Use the original CDC BRFSS age-group code (1–13).",
    "SEXVAR": "Use the original BRFSS code (1 or 2); do not infer identity from the code.",
    "_BMI5CAT": "Use the BRFSS BMI category code (1–4).",
    "EXERANY2": "Original BRFSS exercise response code: 1 or 2.",
    "SMOKE100": "Ever smoked at least 100 cigarettes; original code 1 or 2.",
    "DIABETE4": "Original BRFSS diabetes response code (1–4).",
    "BPHIGH6": "Original BRFSS blood-pressure response code (1–4).",
    "GENHLTH": "Original BRFSS general-health code (1=Excellent through 5=Poor).",
    "PHYSHLTH": "Number of days physical health was not good in the past 30 days.",
    "MENTHLTH": "Number of days mental health was not good in the past 30 days.",
    "EDUCA": "Original BRFSS education code (1–6).",
    "INCOME3": "Original BRFSS income-category code (1–11)."
}


def metric_card(label, value, foot=""):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-foot">{foot}</div></div>',
        unsafe_allow_html=True,
    )


def section(title, subtitle=None):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-sub">{subtitle}</div>', unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading your trained HWRPA model bundle…")
def load_bundle(path_string, modified_time):
    # Only load a joblib file that you generated and trust in your own project repository.
    return joblib.load(path_string)


def read_optional_csv(filename):
    path = BASE / filename
    if path.exists():
        return pd.read_csv(path)
    return None


def get_positive_probability(model, row):
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError(f"Model does not contain positive class 1. Found classes: {classes}")
    return float(model.predict_proba(row)[:, classes.index(1)][0])


# ------------------------------- Sidebar ---------------------------------
with st.sidebar:
    st.markdown("# 🫀 HealthAI")
    st.caption("HWRPA • BRFSS 2025 research prototype")
    st.divider()
    st.markdown("**Navigation**")
    st.markdown("Use the tabs in the main dashboard to explore the model, results and prediction demo.")
    st.divider()
    st.markdown("**Model configuration**")
    st.caption("12 predictors • 4 classifiers • 70:15:15 split")
    st.caption("Positive class: CVDCRHD4 = 1 (Yes)")
    st.divider()
    st.markdown("<span class='small-muted'>Model bundle is loaded from this GitHub repository. No dataset upload is required.</span>", unsafe_allow_html=True)

# ------------------------------- Header ----------------------------------
st.markdown("""
<div class="hero">
 <div class="kicker">Artificial Intelligence · Health Data Analytics · BRFSS 2025</div>
 <h1>AI-Based Health Risk<br>Prediction Platform</h1>
 <p>Hybrid Weighted Risk Prediction Algorithm (HWRPA) — an interactive research dashboard built around the same trained pipelines and validation-weighted ensemble defined in the Google Colab notebook.</p>
</div>
""", unsafe_allow_html=True)

if not BUNDLE_PATH.exists():
    st.error("Trained model bundle not found: hwrpa_model_bundle.joblib")
    st.markdown("""
    <div class="warning-box"><b>One-time setup required:</b> Run notebook Cell 31 in Google Colab, download
    <code>hwrpa_model_bundle.joblib</code>, and add it to the same GitHub repository as <code>app.py</code>.
    Also add the result CSV files created by that cell if you want the evaluation and explainability tabs
    to display your actual notebook outputs. No dataset needs to be uploaded to Streamlit.</div>
    """, unsafe_allow_html=True)
    st.stop()

try:
    bundle = load_bundle(str(BUNDLE_PATH), BUNDLE_PATH.stat().st_mtime)
    models = bundle["models"]
    weights = bundle["weights"]
    features = list(bundle.get("features", FEATURES))
    threshold = float(bundle.get("threshold", 0.5))
    target = bundle.get("target", "CVDCRHD4")
    if features != FEATURES:
        st.warning("The model bundle feature order differs from the dashboard's expected BRFSS feature order. Predictions will use the bundle's feature order.")
except Exception as exc:
    st.error(f"Could not load the trained model bundle: {exc}")
    st.stop()

metrics_df = read_optional_csv("hwrpa_test_metrics.csv")
comparison_df = read_optional_csv("validation_model_comparison.csv")
importance_df = read_optional_csv("feature_importance.csv")

# ------------------------------- Tabs ------------------------------------
overview, predict, performance, explain, methodology = st.tabs([
    "✦ Overview", "◎ Prediction Demo", "▥ Model Performance", "◈ Explainability", "⌘ Methodology"
])

with overview:
    section("Project at a glance", "The deployed dashboard loads your already-trained Colab models. It does not ask for the raw BRFSS dataset or retrain models on every visit.")
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Input predictors", "12", "Selected BRFSS variables")
    with c2: metric_card("ML classifiers", str(len(models)), "Trained in Google Colab")
    with c3: metric_card("Ensemble", "HWRPA", "Validation ROC-AUC weighted")
    with c4: metric_card("Decision threshold", f"{threshold:.2f}", "Selected on validation data")

    st.markdown("<div class='info-box'><b>Seamless deployment:</b> Streamlit reads the trained <code>hwrpa_model_bundle.joblib</code> from the project repository. The raw CDC file is not uploaded by the user and is not processed again by the dashboard.</div>", unsafe_allow_html=True)
    section("The HWRPA pipeline", "A concise view of the workflow implemented in your Colab notebook.")
    stages = st.columns(5)
    stage_info = [("01", "BRFSS 2025", "CDC survey data"), ("02", "Preprocessing", "Code cleaning + imputation"), ("03", "Four models", "LR · Tree · RF · SVM"), ("04", "HWRPA", "Validation AUC weights"), ("05", "Dashboard", "Model-score demonstration")]
    for col, (num, title, desc) in zip(stages, stage_info):
        with col:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>STEP {num}</div><div style='font-size:17px;font-weight:800;margin:9px 0;color:#fff'>{title}</div><div class='metric-foot'>{desc}</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='warning-box'><b>Scientific scope:</b> The notebook classifies the self-reported <code>CVDCRHD4</code> status (1=Yes, 2=No) in a cross-sectional survey. It does <b>not</b> predict future heart disease, provide a clinical risk score, or diagnose a person.</div>", unsafe_allow_html=True)

with predict:
    section("Interactive HWRPA demonstration", "Enter BRFSS-style response codes. The exact fitted pipelines from your Colab model bundle calculate the combined model output.")
    st.markdown("<div class='warning-box'><b>Academic demonstration only.</b> This form is not a medical assessment. Do not enter names, contact details or other identifying information.</div>", unsafe_allow_html=True)
    values = {}
    left, right = st.columns(2)
    for idx, feature in enumerate(features):
        col = left if idx % 2 == 0 else right
        with col:
            values[feature] = st.selectbox(
                LABELS.get(feature, feature), OPTIONS.get(feature, [0]),
                key=f"input_{feature}", help=HELP.get(feature, f"Original BRFSS variable: {feature}")
            )
    if st.button("✦ Generate HWRPA model output", use_container_width=True):
        try:
            row = pd.DataFrame([[values[f] for f in features]], columns=features)
            model_outputs = {name: get_positive_probability(model, row) for name, model in models.items()}
            normalized_weights = {name: float(weights[name]) for name in model_outputs}
            combined = float(sum(normalized_weights[name] * model_outputs[name] for name in model_outputs))
            combined = float(np.clip(combined, 0, 1))
            band = "Higher model-score band" if combined >= threshold else "Lower model-score band"
            st.session_state["last_prediction"] = {"combined": combined, "outputs": model_outputs, "weights": normalized_weights, "band": band}
        except Exception as exc:
            st.error(f"Prediction could not be completed: {exc}")

    result = st.session_state.get("last_prediction")
    if result:
        st.divider()
        section("HWRPA output", "Positive-class model probability for the entered BRFSS-style profile.")
        p1, p2 = st.columns([1, 2])
        with p1:
            metric_card("Combined model probability", f"{result['combined']*100:.1f}%", "Model output · not a clinical risk")
        with p2:
            st.markdown(f"### {result['band']}")
            st.progress(result["combined"])
            st.caption(f"Notebook validation-selected threshold: {threshold:.3f}. This statistical band is illustrative, not clinically validated.")
        pred_df = pd.DataFrame({"Classifier": list(result["outputs"].keys()), "Positive-class probability": list(result["outputs"].values()), "HWRPA weight": [result["weights"][m] for m in result["outputs"]]})
        st.dataframe(pred_df.style.format({"Positive-class probability": "{:.2%}", "HWRPA weight": "{:.4f}"}), use_container_width=True, hide_index=True)
        fig = px.bar(pred_df, x="Classifier", y="Positive-class probability", color="Classifier", text_auto=".1%", template="plotly_dark", title="Individual classifier outputs")
        fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

with performance:
    section("Held-out test performance", "Values below are read from the test-metrics CSV exported by your Colab notebook. No metrics are fabricated.")
    if metrics_df is None or metrics_df.empty:
        st.info("Add the notebook-generated hwrpa_test_metrics.csv to the GitHub repository to display the actual held-out test metrics here.")
    else:
        row = metrics_df.iloc[0]
        aliases = [("Accuracy", "Accuracy"), ("Precision", "Precision"), ("Recall", "Recall"), ("F1-score", "F1"), ("ROC-AUC", "ROC_AUC"), ("Average precision", "Average_Precision")]
        available = [(label, key) for label, key in aliases if key in metrics_df.columns]
        cols = st.columns(min(5, len(available)))
        for i, (label, key) in enumerate(available):
            value = float(row[key])
            with cols[i % len(cols)]: metric_card(label, f"{value*100:.2f}%", "Held-out test set")
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    if comparison_df is not None and not comparison_df.empty:
        section("Validation model comparison", "The four base learners are compared using validation ROC-AUC in the Colab workflow.")
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        if "Model" in comparison_df.columns and "ROC_AUC" in comparison_df.columns:
            fig = px.bar(comparison_df.sort_values("ROC_AUC", ascending=False), x="Model", y="ROC_AUC", color="Model", text_auto=".3f", template="plotly_dark", title="Validation ROC-AUC by classifier")
            fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

    weights_df = pd.DataFrame({"Model": list(weights.keys()), "Validation AUC weight": [float(weights[k]) for k in weights]})
    section("HWRPA ensemble contribution", "Weights are loaded directly from the saved Colab model bundle.")
    st.dataframe(weights_df.style.format({"Validation AUC weight": "{:.4f}"}), use_container_width=True, hide_index=True)
    figw = px.bar(weights_df, x="Model", y="Validation AUC weight", color="Model", text_auto=".3f", template="plotly_dark", title="Contribution of each base learner")
    figw.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(figw, use_container_width=True)

with explain:
    section("Model explainability", "Permutation importance from the notebook describes model reliance, not causation.")
    if importance_df is None or importance_df.empty:
        st.info("Add the notebook-generated feature_importance.csv to the GitHub repository to display the actual feature-importance results.")
    else:
        st.dataframe(importance_df, use_container_width=True, hide_index=True)
        if {"Feature", "Importance_Mean"}.issubset(importance_df.columns):
            plot_df = importance_df.sort_values("Importance_Mean", ascending=True)
            figi = px.bar(plot_df, x="Importance_Mean", y="Feature", orientation="h", error_x="Importance_Std" if "Importance_Std" in plot_df.columns else None, template="plotly_dark", title="Permutation feature importance")
            figi.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_title="Decrease in validation ROC-AUC")
            st.plotly_chart(figi, use_container_width=True)
    st.caption("Importance is an association with model behaviour on the validation data. It does not show that a variable causes coronary heart disease.")

with methodology:
    section("Research methodology", "This dashboard follows the model design and feature definitions in the supplied Google Colab notebook.")
    method = pd.DataFrame([
        ["Dataset", "CDC BRFSS 2025 combined survey file"],
        ["Target", "CVDCRHD4: self-reported coronary heart disease / angina status"],
        ["Target coding", "1 = Yes; 2 = No; other codes excluded"],
        ["Predictors", ", ".join(features)],
        ["Data cleaning", "Numeric conversion; invalid/non-substantive codes set to missing; 88 recoded to 0 for PHYSHLTH and MENTHLTH"],
        ["Missing values", "Most-frequent imputation inside the model preprocessing pipeline"],
        ["Data split", "70% training / 15% validation / 15% test; stratified; random_state=42"],
        ["Base models", "Logistic Regression, Decision Tree, Random Forest, SVM (RBF)"],
        ["Ensemble", "Validation ROC-AUC above 0.50, normalized to sum to one"],
        ["Threshold", "Selected by maximizing validation F1 across thresholds 0.10–0.90"],
        ["Explainability", "Permutation importance using validation ROC-AUC"],
    ], columns=["Component", "Implementation"])
    st.dataframe(method, use_container_width=True, hide_index=True)
    st.markdown("<div class='warning-box'><b>Limitations:</b> BRFSS is cross-sectional and self-reported. This notebook's basic ML workflow does not fully account for complex survey design or survey weights. The output is a classification demonstration of reported status, not a future-disease prediction, clinical diagnosis or treatment recommendation. Response-code meanings should be checked against the official BRFSS 2025 codebook.</div>", unsafe_allow_html=True)

st.divider()
st.markdown("<p style='text-align:center;color:#8298b5;font-size:11px'>AI-Based Health Risk Prediction Platform · HWRPA Research Prototype · CDC BRFSS 2025 · Academic use only</p>", unsafe_allow_html=True)
