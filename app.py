
import io
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(
    page_title="Health Risk Prediction | HWRPA",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

FEATURES = [
    "_AGEG5YR", "SEXVAR", "_BMI5CAT", "EXERANY2", "SMOKE100",
    "DIABETE4", "BPHIGH6", "GENHLTH", "PHYSHLTH", "MENTHLTH",
    "EDUCA", "INCOME3"
]
TARGET = "CVDCRHD4"

LABELS = {
    "_AGEG5YR": "Age group",
    "SEXVAR": "Sex",
    "_BMI5CAT": "BMI category",
    "EXERANY2": "Physical activity",
    "SMOKE100": "Smoking history",
    "DIABETE4": "Diabetes status",
    "BPHIGH6": "High blood pressure",
    "GENHLTH": "General health",
    "PHYSHLTH": "Physical health",
    "MENTHLTH": "Mental health",
    "EDUCA": "Education level",
    "INCOME3": "Income category"
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {
 background:radial-gradient(circle at 8% 5%,rgba(20,184,166,.15),transparent 28%),
 radial-gradient(circle at 95% 4%,rgba(59,130,246,.16),transparent 25%),#0b1220;
 color:#edf4ff;
}
[data-testid="stSidebar"] {background:#111d31;border-right:1px solid #26364d;}
.hero {
 padding:30px 34px;border-radius:22px;margin-bottom:22px;
 background:linear-gradient(115deg,#104456,#15566b 48%,#293d76);
 border:1px solid rgba(148,210,230,.25);box-shadow:0 18px 45px #0005;
}
.kicker {color:#9fe7e1;font-size:12px;font-weight:800;letter-spacing:2px;}
.hero h1 {color:white;font-size:34px;line-height:1.15;margin:9px 0;}
.hero p {color:#d7e8f4;font-size:15px;margin:0;}
.card {
 background:linear-gradient(145deg,#192a41,#111e30);
 border:1px solid #33445b;border-radius:16px;padding:17px 19px;min-height:100px;
}
.card-label {color:#9fb2c9;font-size:11px;font-weight:800;letter-spacing:1px;text-transform:uppercase;}
.card-value {color:#fff;font-size:27px;font-weight:800;margin-top:7px;}
.section {font-size:21px;font-weight:800;color:#eaf4ff;margin:14px 0 4px;}
.sub {font-size:13px;color:#9fb2c9;margin-bottom:14px;}
.note {padding:14px 16px;border-radius:13px;margin:10px 0 18px;
 background:rgba(13,148,136,.12);border:1px solid rgba(45,212,191,.3);color:#ccefeb;}
.warn {padding:14px 16px;border-radius:13px;margin:10px 0 18px;
 background:rgba(180,83,9,.13);border:1px solid rgba(251,191,36,.3);color:#fde7b0;}
div.stButton>button {
 border:0;border-radius:11px;font-weight:800;color:white;
 background:linear-gradient(90deg,#0d9488,#2563eb);
}
</style>
""", unsafe_allow_html=True)


def card(label, value):
    st.markdown(
        f'<div class="card"><div class="card-label">{label}</div>'
        f'<div class="card-value">{value}</div></div>',
        unsafe_allow_html=True
    )


def read_data(file):
    name = file.name.lower()
    raw = file.getvalue()
    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(raw), low_memory=False)
    if name.endswith(".xpt"):
        import pyreadstat
        import tempfile
        # pyreadstat reads SAS transport files reliably from a filesystem path.
        with tempfile.NamedTemporaryFile(suffix=".xpt", delete=True) as temp_file:
            temp_file.write(raw)
            temp_file.flush()
            df, _ = pyreadstat.read_xport(temp_file.name)
        return df
    raise ValueError("Upload a CSV or XPT file.")


def prepare_data(df):
    df = df.copy()
    df.columns = [str(c).strip().upper() for c in df.columns]
    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    df = df[required].copy()
    for col in required:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    return df.dropna(subset=[TARGET])


def preprocessor():
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([("categorical", cat_pipe, FEATURES)])


def models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1200, solver="liblinear", random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42, min_samples_leaf=5),
        "Random Forest": RandomForestClassifier(
            n_estimators=120, min_samples_leaf=2, max_features="sqrt",
            random_state=42, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=120, min_samples_leaf=2, max_features="sqrt",
            random_state=42, n_jobs=-1
        )
    }


def positive_proba(pipe, x, positive_label):
    classes = list(pipe.named_steps["classifier"].classes_)
    if positive_label not in classes:
        raise ValueError("Selected positive class is not available in the fitted model.")
    return pipe.predict_proba(x)[:, classes.index(positive_label)]


def train_ensemble(df, positive_label, limit=None):
    data = df.copy()
    if limit and len(data) > limit:
        # Proportional stratified sampling without relying on newer pandas APIs.
        pieces = []
        proportions = data[TARGET].value_counts(normalize=True)
        for label, proportion in proportions.items():
            group = data[data[TARGET] == label]
            n = max(1, int(round(limit * proportion)))
            pieces.append(group.sample(n=min(n, len(group)), random_state=42))
        data = pd.concat(pieces).sample(frac=1, random_state=42)

    x = data[FEATURES].copy()
    y = data[TARGET].astype(str).copy()
    counts = y.value_counts()
    if positive_label not in counts.index or len(counts) < 2 or counts.min() < 10:
        raise ValueError("Target needs at least two classes, with at least 10 records per class.")

    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=.30, random_state=42, stratify=y
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=.50, random_state=42, stratify=y_temp
    )

    fitted, val_auc = {}, {}
    y_val_bin = (y_val.to_numpy() == positive_label).astype(int)

    for name, estimator in models().items():
        pipe = Pipeline([
            ("preprocessor", preprocessor()),
            ("classifier", estimator)
        ])
        pipe.fit(x_train, y_train)
        fitted[name] = pipe
        p = positive_proba(pipe, x_val, positive_label)
        val_auc[name] = float(roc_auc_score(y_val_bin, p))

    raw_weights = {name: max(score - .5, 0.0) for name, score in val_auc.items()}
    total = sum(raw_weights.values())
    if total == 0:
        weights = {name: 1 / len(raw_weights) for name in raw_weights}
    else:
        weights = {name: value / total for name, value in raw_weights.items()}

    y_test_bin = (y_test.to_numpy() == positive_label).astype(int)
    probs = {name: positive_proba(pipe, x_test, positive_label) for name, pipe in fitted.items()}
    ensemble = sum(weights[name] * probs[name] for name in fitted)
    predicted = (ensemble >= .5).astype(int)

    metrics = {
        "Accuracy": accuracy_score(y_test_bin, predicted),
        "Precision": precision_score(y_test_bin, predicted, zero_division=0),
        "Recall": recall_score(y_test_bin, predicted, zero_division=0),
        "F1-score": f1_score(y_test_bin, predicted, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test_bin, ensemble)
    }
    return {
        "models": fitted, "weights": weights, "validation_auc": val_auc,
        "metrics": metrics, "y_test": y_test_bin, "prob": ensemble,
        "pred": predicted, "positive_label": positive_label,
        "rows": len(data)
    }


def input_form(df):
    st.markdown('<div class="section">Personal health information</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub">Select the raw response codes available in the uploaded BRFSS dataset.</div>',
        unsafe_allow_html=True
    )
    values = {}
    left, right = st.columns(2)
    for i, feature in enumerate(FEATURES):
        with (left if i % 2 == 0 else right):
            choices = sorted(df[feature].dropna().astype(str).unique().tolist())
            if not choices:
                choices = ["Unknown"]
            values[feature] = st.selectbox(
                LABELS[feature], choices, key=f"field_{feature}",
                help=f"Original BRFSS field: {feature}. The app uses the codes found in your uploaded file."
            )
    return pd.DataFrame([values], columns=FEATURES)


# Sidebar
with st.sidebar:
    st.markdown("## 🩺 HealthAI")
    st.caption("HWRPA • Academic research prototype")
    st.divider()
    uploaded = st.file_uploader("Upload BRFSS 2025 dataset", type=["csv", "xpt"])
    sample_mode = st.selectbox("Training size", ["All available records", "Up to 60,000 records"])
    st.divider()
    st.caption("Upload a dataset containing all 12 predictors and CVDCRHD4.")

# Hero
st.markdown("""
<div class="hero">
 <div class="kicker">ARTIFICIAL INTELLIGENCE • HEALTH DATA ANALYTICS</div>
 <h1>AI-Based Health Risk<br>Prediction Platform</h1>
 <p>Hybrid Weighted Risk Prediction Algorithm (HWRPA) — a four-model ensemble
 for classifying self-reported coronary heart disease status in BRFSS survey data.</p>
</div>
""", unsafe_allow_html=True)

overview, evaluation, prediction, about = st.tabs(
    ["✦ Overview", "⚙ Model & Evaluation", "♡ Prediction Demo", "ⓘ About"]
)

if uploaded is None:
    with overview:
        st.markdown('<div class="note">Upload your BRFSS 2025 CSV or XPT file from the sidebar. '
                    'The application will validate the required fields, train four classifiers, '
                    'calculate HWRPA weights and evaluate the ensemble.</div>', unsafe_allow_html=True)
        a, b, c = st.columns(3)
        with a: card("Input features", "12")
        with b: card("ML classifiers", "04")
        with c: card("Proposed ensemble", "HWRPA")
        st.markdown("**Required dataset fields**")
        st.code(", ".join(FEATURES + [TARGET]))
    with evaluation:
        st.info("Upload the dataset to view model evaluation.")
    with prediction:
        st.info("Upload the dataset and train the models before using the prediction demo.")
    with about:
        st.write("The prototype uses BRFSS 2025 survey variables and classifies the CVDCRHD4 self-reported status.")
else:
    try:
        source_df = read_data(uploaded)
        df = prepare_data(source_df)
        target_labels = sorted(df[TARGET].dropna().astype(str).unique().tolist())
        if len(target_labels) < 2:
            raise ValueError("CVDCRHD4 must contain at least two target classes.")

        with st.sidebar:
            st.markdown("### Target configuration")
            positive_label = st.selectbox(
                "Select positive CVDCRHD4 code",
                target_labels,
                help="Choose the positive response code according to the BRFSS 2025 codebook and your project target coding."
            )
            train_button = st.button("🚀 Train HWRPA models", use_container_width=True)

        with overview:
            c1, c2, c3 = st.columns(3)
            with c1: card("Usable records", f"{len(df):,}")
            with c2: card("Selected predictors", "12")
            with c3: card("Target categories", str(len(target_labels)))
            st.markdown('<div class="section">Dataset validation</div>', unsafe_allow_html=True)
            st.success("All required columns were found.")
            summary = pd.DataFrame({
                "Variable": FEATURES + [TARGET],
                "Meaning": [LABELS.get(f, "Classification target") for f in FEATURES] + ["CVDCRHD4 target"],
                "Missing": [int(df[f].isna().sum()) for f in FEATURES + [TARGET]],
                "Unique values": [int(df[f].nunique()) for f in FEATURES + [TARGET]]
            })
            st.dataframe(summary, use_container_width=True, hide_index=True)
            st.markdown(
                '<div class="warn"><b>Important:</b> Verify the selected positive target code using '
                'the official BRFSS 2025 codebook. The app intentionally does not guess the meaning '
                'of numeric survey response codes.</div>', unsafe_allow_html=True
            )

        if train_button:
            with st.spinner("Training Logistic Regression, Decision Tree, Random Forest and Extra Trees..."):
                limit = 60000 if sample_mode.startswith("Up to") else None
                result = train_ensemble(df, positive_label, limit)
                st.session_state["hwrpa_result"] = result
                st.session_state["hwrpa_df"] = df
                st.session_state["hwrpa_label"] = positive_label
            st.success("HWRPA training completed.")

        result = st.session_state.get("hwrpa_result")
        active_label = st.session_state.get("hwrpa_label")

        with evaluation:
            if result is None:
                st.info("Choose the positive target code and click **Train HWRPA models** in the sidebar.")
            else:
                st.markdown('<div class="section">HWRPA test-set performance</div>', unsafe_allow_html=True)
                st.caption(f"Positive class: {active_label} • Records used: {result['rows']:,} • Split: 70% / 15% / 15%")
                metric_cols = st.columns(5)
                for col, (name, value) in zip(metric_cols, result["metrics"].items()):
                    with col:
                        card(name, f"{value * 100:.2f}%")

                st.markdown('<div class="section">Validation-based model weights</div>', unsafe_allow_html=True)
                weights_df = pd.DataFrame({
                    "Model": list(result["weights"].keys()),
                    "Validation ROC-AUC": [result["validation_auc"][m] for m in result["weights"]],
                    "HWRPA weight": list(result["weights"].values())
                })
                st.dataframe(weights_df.style.format({
                    "Validation ROC-AUC": "{:.4f}", "HWRPA weight": "{:.4f}"
                }), use_container_width=True, hide_index=True)
                fig_w = px.bar(weights_df, x="Model", y="HWRPA weight", color="Model",
                               title="Contribution of each classifier", template="plotly_dark")
                fig_w.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_w, use_container_width=True)

                cm = confusion_matrix(result["y_test"], result["pred"], labels=[0, 1])
                cm_df = pd.DataFrame(cm, index=["Actual negative", "Actual positive"],
                                     columns=["Predicted negative", "Predicted positive"])
                fig_cm = px.imshow(cm_df, text_auto=True, aspect="auto",
                                   title="HWRPA confusion matrix", color_continuous_scale="Teal",
                                   template="plotly_dark")
                fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_cm, use_container_width=True)

                fpr, tpr, _ = roc_curve(result["y_test"], result["prob"])
                roc_df = pd.DataFrame({"False positive rate": fpr, "True positive rate": tpr})
                fig_roc = px.line(roc_df, x="False positive rate", y="True positive rate",
                                  title=f"ROC curve • AUC {result['metrics']['ROC-AUC']:.3f}",
                                  template="plotly_dark")
                fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                                  line=dict(dash="dash", color="#94a3b8"))
                fig_roc.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_roc, use_container_width=True)

        with prediction:
            if result is None:
                st.info("Train the models first to activate the prediction demo.")
            else:
                st.markdown(
                    '<div class="warn">Research demonstration only. This predicts a BRFSS-style '
                    'self-reported classification label. It does not diagnose heart disease or '
                    'estimate future disease risk.</div>', unsafe_allow_html=True
                )
                user_input = input_form(df)
                if st.button("✨ Analyse survey profile", key="run_prediction"):
                    try:
                        individual = {
                            name: float(positive_proba(pipe, user_input, active_label)[0])
                            for name, pipe in result["models"].items()
                        }
                        final_probability = sum(
                            result["weights"][name] * individual[name] for name in individual
                        )
                        st.divider()
                        st.markdown('<div class="section">HWRPA model output</div>', unsafe_allow_html=True)
                        p1, p2 = st.columns([1, 2])
                        with p1:
                            card("Combined positive-class probability", f"{final_probability * 100:.1f}%")
                        with p2:
                            if final_probability >= .5:
                                st.info(f"Model output: selected positive class ({active_label}).")
                            else:
                                st.info("Model output: other target class.")
                            st.progress(float(np.clip(final_probability, 0, 1)))
                        st.markdown("**Individual classifier outputs**")
                        probs_df = pd.DataFrame({
                            "Classifier": list(individual.keys()),
                            "Positive-class probability": list(individual.values()),
                            "HWRPA weight": [result["weights"][m] for m in individual]
                        })
                        st.dataframe(probs_df.style.format({
                            "Positive-class probability": "{:.2%}", "HWRPA weight": "{:.4f}"
                        }), use_container_width=True, hide_index=True)
                    except Exception as exc:
                        st.error(f"Prediction could not be completed: {exc}")

        with about:
            st.markdown('<div class="section">Project details</div>', unsafe_allow_html=True)
            st.write("**Dataset:** CDC BRFSS 2025.")
            st.write("**Target:** CVDCRHD4 — self-reported angina / coronary heart disease status.")
            st.write("**Predictors:** 12 selected demographic, lifestyle and health-status variables.")
            st.write("**Classifiers:** Logistic Regression, Decision Tree, Random Forest and Extra Trees.")
            st.write("**Ensemble:** HWRPA using normalized validation ROC-AUC-based weights.")
            st.write("**Explainability method in the project:** Permutation feature importance.")
            st.markdown(
                '<div class="warn"><b>Medical limitation:</b> BRFSS is cross-sectional and self-reported. '
                'This is an academic research prototype, not a medical device. Do not use it for '
                'diagnosis, treatment or emergency decisions.</div>', unsafe_allow_html=True
            )

    except Exception as exc:
        st.error(f"Dataset could not be prepared: {exc}")
        st.info("Check the file format and confirm that it contains all 12 predictors plus CVDCRHD4.")

st.divider()
st.markdown(
    '<p style="text-align:center;color:#91a4bd;font-size:12px;">AI-Based Health Risk Prediction Platform '
    '• HWRPA Research Prototype • BRFSS 2025 • Academic demonstration only</p>',
    unsafe_allow_html=True
)
