# 🫀 Cardiac Risk Prediction — Explainable Artificial Intelligence

> An end-to-end machine learning framework for biomarker-driven cardiovascular risk stratification with explainability using SHAP and a RAG-based patient report generator.

---

## 🌐 Web App
**[👉 Click here to try the Live Demo](https://huggingface.co/spaces/Ushaswiii/Cardiac-Risk-Prediction)**

---

## 📌 Description

Cardiovascular diseases (CVDs) remain the leading cause of death worldwide. Early and accurate risk stratification is critical for timely intervention. This project introduces an explainable machine learning framework that predicts a patient's cardiovascular risk level — **Low**, **Moderate**, or **High** — using laboratory biomarker data.

The framework combines:
- A **CatBoost classifier** tuned with **Optuna (Bayesian optimization)** for high-accuracy multiclass prediction
- **SHAP (SHapley Additive exPlanations)** for global and per-patient explainability
- A **RAG-based explanation module** that generates structured, human-readable patient risk reports
- An interactive **Streamlit dashboard** for real-world clinical decision support

---

## 🎯 Objective

Given a patient's blood test results, predict their cardiovascular risk category and explain **which biomarkers drove the prediction** and **by how much**.

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| Source | [Kaggle — Biomarkers for Cardiovascular Risk (US Hospitals)](https://www.kaggle.com/datasets/taruneshburman/biomarkers-for-cardiovascular-risk-us-hospitals) |
| Patients | 1,200 |
| Features | 24 standardized biomarker features (z-scores) |
| Target | `composite_risk_score` → Low (0) / Moderate (1) / High (2) |
| Split | 70% Train / 15% Validation / 15% Test (stratified) |

### Key Features

| Category | Features |
|----------|---------|
| Raw Biomarkers | `troponin_level`, `c_reactive_protein`, `fasting_glucose`, `creatinine`, `hdl_cholesterol`, `ldl_cholesterol`, `triglycerides`, `wbc_count`, `platelet_count`, `ekg_result`, `recent_hospital_visit` |
| Engineered Ratios | `crp_hdl_ratio`, `troponin_crp_product`, `troponin_hdl_ratio`, `troponin_ldl_ratio`, `tg_hdl_ratio`, `creatinine_wbc_ratio`, `non_hdl`, `tyg_index`, `aip`, and more |

---

## 🧠 Methodology

1. **Preprocessing** — Standardization (z-scores), label encoding, stratified splitting
2. **Feature Engineering** — 13 derived biomarker ratios and composite indices
3. **Classification** — CatBoost multiclass classifier (`multi:softprob`)
4. **Hyperparameter Tuning** — Optuna with Tree-structured Parzen Estimator (TPE), 50 trials
5. **Class Imbalance** — Class weights + probability threshold calibration (t1=0.4, t2=0.12)
6. **Explainability** — SHAP TreeExplainer for global + per-patient feature attribution
7. **RAG Report** — Structured patient-level explanation from SHAP feature contributions

---

## 📈 Results

### Classification Report (CatBoost + Optuna)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Low Risk (0) | 0.96 | 1.00 | 0.98 | 51 |
| Moderate Risk (1) | 0.99 | 0.95 | 0.97 | 87 |
| High Risk (2) | 0.95 | 0.98 | 0.96 | 42 |
| **Overall** | **0.97** | **0.97** | **0.97** | **180** |

### Model Comparison

| Model | Accuracy | F1-Score | AUC |
|-------|----------|----------|-----|
| TabNet | 0.76 | 0.74 | 0.893 |
| Extra Trees | 0.82 | 0.80 | 0.947 |
| Random Forest | 0.92 | 0.92 | 0.987 |
| Histogram Gradient Boosting | 0.92 | 0.92 | 0.991 |
| XGBoost | 0.94 | 0.94 | 0.997 |
| **CatBoost + Optuna (Proposed)** | **0.97** | **0.97** | **0.997** |

---

## 🔍 Explainability (SHAP)

- **Top global features:** `c_reactive_protein` and `troponin_level` dominate predictions
- **Clinical alignment:** CRP (inflammation) and troponin (cardiac injury) are the primary real-world cardiac risk biomarkers
- **Per-patient SHAP:** Each prediction shows which biomarkers pushed toward or away from the predicted risk class
- **RAG Report:** Human-readable summary with predicted class, confidence score, and top contributing biomarkers

---

## 🖥️ Dashboard Features

- 📁 Upload patient lab report (CSV)
- 👤 Enter patient name and age
- 🔢 View and edit 8 key lab values
- 🎯 Predict risk category with confidence score
- 📊 Probability bar chart across all 3 classes
- 🔴 Per-patient SHAP factor chart
- 📋 Structured patient risk assessment report with clinical disclaimer

---

## 🚀 Setup & Run Locally

```bash
# Clone the repo
git clone https://github.com/ushaswi09/Cardiac-Risk-Prediction-Explainable-Artificial-Intelligence.git
cd Cardiac-Risk-Prediction-Explainable-Artificial-Intelligence

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install catboost xgboost shap scikit-learn pandas numpy matplotlib streamlit joblib

# Run dashboard
streamlit run dashboard/app.py
```

### Test with sample patient
```bash
python3 -c "
import pandas as pd
df = pd.read_csv('data/final_standardized_dataset(2).csv')
df.drop(columns=['patient_id','composite_risk_score','composite_risk_category'], errors='ignore').head(1).to_csv('sample_patient.csv', index=False)
"
```
Upload `sample_patient.csv` in the dashboard to test.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| CatBoost | Primary classifier |
| XGBoost | Baseline model |
| Optuna | Hyperparameter tuning |
| SHAP | Explainability |
| scikit-learn | Preprocessing & metrics |
| Streamlit | Dashboard |
| Hugging Face Spaces | Deployment |

---

## ⚠️ Clinical Note

This framework is intended for **decision-support purposes only**. All outputs are probabilistic estimates from a machine learning model and **must not replace comprehensive clinical evaluation**. All medical decisions should be made by a qualified healthcare professional.
