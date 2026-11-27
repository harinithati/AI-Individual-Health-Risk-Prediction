
import os
import sys
import joblib
import streamlit as st
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="AI Health Risk Prediction",
    page_icon="🩺",
    layout="wide"
)
import os
import hashlib
import streamlit as st
import sklearn
import joblib
import sys

MODEL_PATH = "HWRPA_Model.pkl"

st.write("### Model Diagnostic Information")

st.write("Python version:", sys.version)
st.write("Scikit-learn version:", sklearn.__version__)
st.write("Joblib version:", joblib.__version__)

if os.path.exists(MODEL_PATH):

    with open(MODEL_PATH, "rb") as f:
        model_bytes = f.read()

    st.write("Model file size:", len(model_bytes))
    st.write("Model header:", repr(model_bytes[:16]))
    st.write(
        "Model SHA256:",
        hashlib.sha256(model_bytes).hexdigest()
    )

else:
    st.error("HWRPA_Model.pkl not found!")
    st.stop()
# Load trained model
bundle = joblib.load("HWRPA_Model.pkl")

models = bundle["models"]
weights = bundle["weights"]
features = bundle["features"]

st.title("AI-Based Individual Health Risk Prediction Platform")

st.write(
    "An AI-based research prototype for individual health assessment "
    "using machine-learning techniques."
)

st.warning(
    "Academic demonstration only. This application does not provide "
    "medical diagnosis or clinically validated future-risk predictions."
)

st.header("Enter Individual Health Information")

# BRFSS-coded input options
options = {
    "_AGEG5YR": list(range(1, 14)),
    "SEXVAR": [1, 2],
    "_BMI5CAT": [1, 2, 3, 4],
    "EXERANY2": [1, 2],
    "SMOKE100": [1, 2],
    "DIABETE4": [1, 2, 3, 4],
    "BPHIGH6": [1, 2, 3, 4],
    "GENHLTH": [1, 2, 3, 4, 5],
    "EDUCA": [1, 2, 3, 4, 5, 6],
    "INCOME3": list(range(1, 12))
}

labels = {
    "_AGEG5YR": "Age group (BRFSS code)",
    "SEXVAR": "Sex (BRFSS code)",
    "_BMI5CAT": "BMI category (BRFSS code)",
    "EXERANY2": "Exercise response",
    "SMOKE100": "Lifetime smoking response",
    "DIABETE4": "Diabetes response code",
    "BPHIGH6": "High blood pressure response code",
    "GENHLTH": "General health (1=Excellent, 5=Poor)",
    "PHYSHLTH": "Physical unhealthy days (0–30)",
    "MENTHLTH": "Mental unhealthy days (0–30)",
    "EDUCA": "Education code",
    "INCOME3": "Income category code"
}

input_values = {}

with st.form("health_form"):

    for feature in features:

        label = labels.get(feature, feature)

        if feature in options:
            input_values[feature] = st.selectbox(
                label,
                options[feature]
            )

        elif feature in ["PHYSHLTH", "MENTHLTH"]:
            input_values[feature] = st.number_input(
                label,
                min_value=0,
                max_value=30,
                value=0
            )

        else:
            st.error(f"Input mapping not configured for {feature}.")
            st.stop()

    submitted = st.form_submit_button("Generate Assessment")

if submitted:

    input_df = pd.DataFrame(
        [input_values],
        columns=features
    )

    probability = 0.0

    for name, weight in weights.items():

        model_probability = models[name].predict_proba(
            input_df
        )[0, 1]

        probability += weight * model_probability

    st.header("Assessment Results")

    st.metric(
        "Model-estimated probability",
        f"{probability * 100:.2f}%"
    )

    st.info(
        "This probability estimates the selected BRFSS-reported "
        "health outcome. It is not a clinical risk score."
    )

    st.subheader("Contributing Input Information")
    st.dataframe(input_df, use_container_width=True)

    st.success("Assessment generated successfully.")
