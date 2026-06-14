import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import streamlit as st

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

OUT_DIR = "outputs/catboost_optuna"
CB_MODEL_PATH = os.path.join(OUT_DIR, "catboost_best_model.cbm")
XGB_MODEL_PATH = "outputs/xgb_model.json"
FEATURES_PATH = "outputs/xgb_feature_list.pkl"

RISK_LABELS = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk"}
RISK_COLORS = {0: "green", 1: "orange", 2: "red"}

KEY_FEATURES = [
    "c_reactive_protein",
    "troponin_level",
    "fasting_glucose",
    "creatinine",
    "crp_hdl_ratio",
    "troponin_crp_product",
    "platelet_count",
    "wbc_count",
]

st.set_page_config(page_title="Cardiac Risk Predictor", layout="centered")
st.title("Cardiac Risk Prediction")
st.caption("An explainable AI framework for biomarker-driven cardiovascular risk stratification.")

@st.cache_resource
def load_model():
    features = joblib.load(FEATURES_PATH)
    if CATBOOST_AVAILABLE and os.path.exists(CB_MODEL_PATH):
        model_type = "catboost"
        model = CatBoostClassifier()
        model.load_model(CB_MODEL_PATH)
    else:
        model_type = "xgboost"
        model = xgb.Booster()
        model.load_model(XGB_MODEL_PATH)
    return model, features, model_type

model, features, model_type = load_model()
st.caption(f"Model: {'CatBoost (Optuna-tuned)' if model_type == 'catboost' else 'XGBoost'} — 97% accuracy, AUC 0.9973" if model_type == "catboost" else "Model: XGBoost — 91% accuracy, AUC 0.993")

st.markdown("---")

# Patient Info
st.subheader("Patient Info")
c1, c2 = st.columns(2)
with c1:
    patient_name = st.text_input("Patient Name", value="")
with c2:
    patient_age = st.number_input("Age", min_value=0, max_value=120, value=45, step=1)

# Upload
st.subheader("Upload Lab Report (CSV)")
uploaded = st.file_uploader("Upload a CSV with patient lab values (one row)", type="csv")

input_vals = {f: 0.0 for f in features}

if uploaded is not None:
    up_df = pd.read_csv(uploaded)
    row = up_df.iloc[0]
    for f in features:
        if f in row:
            input_vals[f] = float(row[f])
    st.success("File loaded. Values auto-filled below (editable).")

# Key lab inputs
st.subheader("Key Lab Values")
st.caption("Standardized z-scores: 0 = population average, positive = above average, negative = below average.")

cols = st.columns(2)
for i, feat in enumerate(KEY_FEATURES):
    if feat in features:
        with cols[i % 2]:
            input_vals[feat] = st.number_input(
                feat, value=float(input_vals[feat]), format="%.4f"
            )

st.markdown("---")

if st.button("Predict Risk", type="primary"):
    X_input = pd.DataFrame([input_vals])[features]

    # Predict
    if model_type == "catboost":
        proba = model.predict_proba(X_input)[0]
        t2 = 0.12
        t1 = 0.4
        if proba[2] >= t2:
            pred_class = 2
        elif proba[1] >= t1:
            pred_class = 1
        else:
            pred_class = 0
    else:
        dmatrix = xgb.DMatrix(X_input)
        proba = model.predict(dmatrix)[0]
        if proba[2] >= 0.12:
            pred_class = 2
        elif proba[1] >= 0.4:
            pred_class = 1
        else:
            pred_class = 0

    confidence = proba[pred_class] * 100
    name_display = patient_name if patient_name else "Patient"

    # Result
    st.subheader(f"Result for {name_display} (Age {patient_age})")
    st.markdown(f"### Predicted Category: :{RISK_COLORS[pred_class]}[{RISK_LABELS[pred_class]}]")
    st.markdown(f"**Prediction Confidence: {confidence:.2f}%**")

    proba_df = pd.DataFrame({
        "Risk Category": [RISK_LABELS[0], RISK_LABELS[1], RISK_LABELS[2]],
        "Probability": proba
    })
    st.bar_chart(proba_df.set_index("Risk Category"))

    # SHAP
    st.markdown("---")
    st.subheader("Key Factors Behind This Prediction")

    if model_type == "catboost":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_input)
        if isinstance(shap_values, list):
            sv = shap_values[pred_class][0]
        else:
            sv = shap_values[0, :, pred_class]
    else:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_input)
        if isinstance(shap_values, list):
            sv = shap_values[pred_class][0]
        else:
            sv = shap_values[0, :, pred_class]

    shap_df = pd.DataFrame({
        "feature": features,
        "shap_value": sv,
        "patient_value": X_input.iloc[0].values
    })
    shap_df["abs_shap"] = shap_df["shap_value"].abs()
    shap_df = shap_df.sort_values("abs_shap", ascending=False).head(8)

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["crimson" if v > 0 else "steelblue" for v in shap_df["shap_value"]]
    ax.barh(shap_df["feature"][::-1], shap_df["shap_value"][::-1], color=colors[::-1])
    ax.set_xlabel("SHAP value")
    ax.axvline(0, color="gray", linewidth=0.8)
    st.pyplot(fig)
    st.caption("Red = pushes toward predicted risk. Blue = pushes away from it.")

    # RAG-style explanation
    st.markdown("---")
    st.subheader("Patient Risk Assessment Report")

    top_features = shap_df.head(4)
    report_lines = [
        f"**Predicted Risk Category:** {RISK_LABELS[pred_class]}",
        f"**Prediction Confidence:** {confidence:.2f}%",
        "",
        "**Primary Contributing Biomarkers:**",
    ]
    for i, (_, row) in enumerate(top_features.iterrows(), 1):
        direction = "positively contributed" if row["shap_value"] > 0 else "negatively contributed"
        report_lines.append(f"{i}. **{row['feature']}** ({direction}; SHAP impact = {abs(row['shap_value']):.4f})")

    report_lines += [
        "",
        "**Model Interpretation:**",
        "The predicted risk classification is driven by the cumulative effect of the above biomarkers. "
        "Features with larger absolute SHAP values exert stronger influence on the final model decision. "
        "Comparatively weaker feature contributions reduced the likelihood of alternative risk categories.",
        "",
        "> ⚠️ **Clinical Note:** This output represents a probabilistic estimate generated by a machine learning model. "
        "It is intended for decision-support purposes only and must not replace comprehensive clinical evaluation. "
        "All medical decisions should be made by a qualified healthcare professional."
    ]

    for line in report_lines:
        st.markdown(line)
