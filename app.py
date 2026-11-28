import os
import glob
import joblib
import pandas as pd
import streamlit as st

# ---------------- PAGE SETUP ----------------
st.set_page_config(
    page_title="HealthAI | HWRPA",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_FILE = "HWRPA_Model.pkl"

# ---------------- SIMPLE STYLING ----------------
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0b1220 0%, #111c31 55%, #17233b 100%); color: #f4f7fb; }
    [data-testid="stSidebar"] { background: #101a2b; }
    .hero {
        padding: 26px 30px; border-radius: 20px;
        background: linear-gradient(110deg, #173d55, #293c70);
        border: 1px solid rgba(160,210,255,.25); margin-bottom: 18px;
    }
    .hero h1 { color: white; margin: 0; font-size: 2.15rem; }
    .hero p { color: #d6e6f6; margin: 8px 0 0 0; font-size: 1rem; }
    .card {
        background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.12);
        border-radius: 16px; padding: 18px 20px; margin-bottom: 12px;
    }
    .muted { color: #b9c7d9; font-size: .92rem; }
    div.stButton > button {
        background: linear-gradient(90deg,#18a6a6,#4778d8); color: white;
        border: 0; border-radius: 10px; padding: .65rem 1.2rem; font-weight: 700;
    }
    div.stButton > button:hover { border: 1px solid #9ed9ff; color: white; }
    [data-testid="stMetric"] {
        background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.12);
        padding: 14px; border-radius: 14px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- MODEL LOADING ----------------
@st.cache_resource(show_spinner="Loading your trained HWRPA model...")
def load_model():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            f"{MODEL_FILE} was not found. Put it in the same GitHub folder as app.py."
        )
    bundle = joblib.load(MODEL_FILE)
    if not isinstance(bundle, dict):
        raise TypeError("The saved model file is not the expected HWRPA dictionary.")
    required = {"models", "weights", "features", "threshold"}
    missing = required.difference(bundle.keys())
    if missing:
        raise KeyError(f"Model file is missing these fields: {sorted(missing)}")
    return bundle

# ---------------- VISUALS ----------------
def find_chart(*patterns):
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    return None

def show_chart(title, *patterns):
    path = find_chart(*patterns)
    st.markdown(f"#### {title}")
    if path:
        st.image(path, use_container_width=True)
    else:
        st.info("Chart file not found yet. Upload the corresponding PNG from Colab beside app.py.")

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
  <h1>🫀 HealthAI — Individual Health Risk Research Platform</h1>
  <p>HWRPA • Validation-weighted machine-learning ensemble • Interactive research dashboard</p>
</div>
""", unsafe_allow_html=True)

st.warning(
    "Research prototype only. This model classifies a self-reported BRFSS health-status outcome; "
    "it does not predict future disease, provide a diagnosis, or replace medical advice."
)

# ---------------- LOAD BUNDLE ----------------
try:
    bundle = load_model()
    models = bundle["models"]
    weights = bundle["weights"]
    features = bundle["features"]
    threshold = float(bundle["threshold"])
except Exception as exc:
    st.error("The trained model could not be loaded.")
    st.code(f"{type(exc).__name__}: {exc}")
    st.markdown(
        "**Deployment check:** use Python 3.11 and the pinned requirements supplied with this app. "
        "Keep `HWRPA_Model.pkl` in the same repository folder as `app.py`."
    )
    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.title("HealthAI Navigation")
page = st.sidebar.radio(
    "Choose a section",
    ["Overview", "Make a Prediction", "Model Insights", "About the Project"],
)
st.sidebar.markdown("---")
st.sidebar.caption("HWRPA research prototype | CDC BRFSS 2025")

# ---------------- OVERVIEW ----------------
if page == "Overview":
    st.subheader("Project Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Models in ensemble", len(models))
    c2.metric("Input features", len(features))
    c3.metric("Decision threshold", f"{threshold:.2f}")
    st.markdown("""
    <div class="card">
      <h3>How the platform works</h3>
      <p class="muted">The dashboard loads the trained pipelines created in Google Colab.
      It combines their positive-class probabilities using the saved validation-based weights.
      The original dataset is not uploaded or processed by this Streamlit app.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Ensemble model weights")
    weight_df = pd.DataFrame({
        "Model": list(weights.keys()),
        "Ensemble weight": [float(v) for v in weights.values()]
    }).sort_values("Ensemble weight", ascending=False)
    st.dataframe(weight_df, use_container_width=True, hide_index=True)
    st.bar_chart(weight_df.set_index("Model"))

# ---------------- PREDICTION ----------------
elif page == "Make a Prediction":
    st.subheader("Interactive Model Estimate")
    st.write("Enter the BRFSS-coded values below. The input controls use the feature coding from the Colab notebook.")

    options = {
        "_AGEG5YR": list(range(1, 14)),
        "SEXVAR": [1, 2],
        "_BMI5CAT": [1, 2, 3, 4],
        "EXERANY2": [1, 2],
        "SMOKE100": [1, 2],
        "DIABETE4": [1, 2, 3, 4],
        "BPHIGH6": [1, 2, 3, 4],
        "GENHLTH": [1, 2, 3, 4, 5],
        "PHYSHLTH": list(range(31)),
        "MENTHLTH": list(range(31)),
        "EDUCA": list(range(1, 7)),
        "INCOME3": list(range(1, 12)),
    }
    labels = {
        "_AGEG5YR": "Age group code",
        "SEXVAR": "Sex code",
        "_BMI5CAT": "BMI category code",
        "EXERANY2": "Exercise in past 30 days (1=Yes, 2=No)",
        "SMOKE100": "Smoked 100+ cigarettes (1=Yes, 2=No)",
        "DIABETE4": "Diabetes response code",
        "BPHIGH6": "High blood pressure response code",
        "GENHLTH": "General health (1=Excellent to 5=Poor)",
        "PHYSHLTH": "Physical unhealthy days (0–30)",
        "MENTHLTH": "Mental unhealthy days (0–30)",
        "EDUCA": "Education code",
        "INCOME3": "Income category code",
    }

    with st.form("prediction_form"):
        cols = st.columns(2)
        values = {}
        for i, feature in enumerate(features):
            choices = options.get(feature, [0])
            with cols[i % 2]:
                values[feature] = st.selectbox(
                    labels.get(feature, feature),
                    choices,
                    key=f"input_{feature}",
                )
        submitted = st.form_submit_button("Generate model estimate")

    if submitted:
        try:
            row = pd.DataFrame([[values[f] for f in features]], columns=features)
            probability = 0.0
            for model_name, weight in weights.items():
                model = models[model_name]
                model_probability = float(model.predict_proba(row)[0, 1])
                probability += float(weight) * model_probability

            probability = max(0.0, min(1.0, probability))
            score = probability * 100

            if probability < threshold * 0.5:
                band = "Lower model-score band"
            elif probability < threshold:
                band = "Moderate model-score band"
            else:
                band = "Higher model-score band"

            st.success("Model estimate generated.")
            a, b = st.columns(2)
            a.metric("Model-estimated score", f"{score:.1f}/100")
            b.metric("Relative score band", band)
            st.progress(probability)
            st.caption(
                "This is a model score for the dataset's self-reported outcome, "
                "not a personal medical risk estimate or clinical threshold."
            )
        except Exception as exc:
            st.error("Prediction could not be generated for these inputs.")
            st.code(f"{type(exc).__name__}: {exc}")

# ---------------- MODEL INSIGHTS ----------------
elif page == "Model Insights":
    st.subheader("Model Performance & Explainability")
    st.markdown("### Evaluation visuals from Google Colab")
    left, right = st.columns(2)
    with left:
        show_chart("ROC Curve", "HWRPA_ROC_Curve.png", "hwrpa_roc_curve.png")
        show_chart("Confusion Matrix", "HWRPA_Confusion_Matrix.png", "hwrpa_confusion_matrix.png")
    with right:
        show_chart("Model Performance", "HWRPA_Performance_Graph.png", "model_comparison.png")
        show_chart("Feature Importance", "Feature_Importance.png", "feature_importance.png")

    comparison_csv = find_chart("validation_model_comparison.csv", "Model_Performance_Compar*.csv")
    metrics_csv = find_chart("hwrpa_test_metrics.csv")
    importance_csv = find_chart("feature_importance.csv")

    if comparison_csv:
        st.markdown("#### Validation model comparison")
        st.dataframe(pd.read_csv(comparison_csv), use_container_width=True)
    if metrics_csv:
        st.markdown("#### HWRPA test metrics")
        st.dataframe(pd.read_csv(metrics_csv), use_container_width=True)
    if importance_csv:
        st.markdown("#### Feature importance results")
        st.dataframe(pd.read_csv(importance_csv), use_container_width=True)

# ---------------- ABOUT ----------------
else:
    st.subheader("About this Research Prototype")
    st.markdown("""
    **Dataset:** CDC Behavioral Risk Factor Surveillance System (BRFSS) 2025.

    **Target:** `CVDCRHD4`, converted to a binary label for reported coronary heart disease/angina status.

    **Input variables:** the 12 coded predictors selected in the Colab notebook.

    **Ensemble:** validation ROC-AUC-based weights are applied to the positive-class probabilities
    from the saved trained models.

    **Important limitations:**
    - The source data are cross-sectional and self-reported.
    - The task is classification of reported health status, not prediction of future disease.
    - The model is an academic prototype and is not clinically validated.
    - Score bands are illustrative statistical groupings, not medical risk categories.
    """)
    st.markdown("#### Loaded model components")
    st.write(", ".join(models.keys()))
    st.write("Feature order:", ", ".join(features))
    st.write("Target:", bundle.get("target", "Not stored in model bundle"))
