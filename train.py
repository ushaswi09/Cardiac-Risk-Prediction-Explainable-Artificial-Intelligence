# XGBoost + SHAP Training Pipeline (VS Code Jupyter Compatible)

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import joblib

# ---------------- CONFIG ----------------
os.chdir("/Users/ushaswikaruturi/Desktop/project")
DATA_PATH = "data/final_standardized_dataset(2).csv"

OUT_DIR = "outputs"
MODEL_OUT = os.path.join(OUT_DIR, "xgb_model.json")
FEATURES_OUT = os.path.join(OUT_DIR, "xgb_feature_list.pkl")
SHAP_PLOT_OUT = os.path.join(OUT_DIR, "shap_summary.png")
PRED_OUT = os.path.join(OUT_DIR, "xgb_test_predictions.csv")

RANDOM_STATE = 42
# ----------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

# 1. Load dataset
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print("Loaded:", DATA_PATH, "shape:", df.shape)

# 2. Create target
if "composite_risk_score" not in df.columns:
    raise KeyError("composite_risk_score missing")

def map_category(s):
    if s == 0:
        return 0
    elif s == 1:
        return 1
    else:
        return 2

df["risk_category"] = df["composite_risk_score"].apply(map_category)

print("Overall class distribution:")
print(df["risk_category"].value_counts())

assert set(df["risk_category"].unique()) == {0, 1, 2}, \
    "ERROR: Dataset does not contain all 3 classes"

# 3. Feature selection
exclude = ["patient_id", "composite_risk_score", "composite_risk_category", "risk_category"]
features = [c for c in df.columns if c not in exclude]
print("Number of features:", len(features))

for c in features:
    if df[c].dtype == "object":
        try:
            df[c] = pd.to_numeric(df[c])
        except:
            df[c] = LabelEncoder().fit_transform(df[c].astype(str))

X = df[features]
y = df["risk_category"]

# 4. Stratified split (70/15/15)
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE
)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.17647,
    stratify=y_trainval, random_state=RANDOM_STATE
)

print("Train classes:", np.unique(y_train))
print("Val classes:", np.unique(y_val))
print("Test classes:", np.unique(y_test))

assert set(y_train.unique()) == {0,1,2}, "Class missing in TRAIN"
assert set(y_val.unique()) == {0,1,2}, "Class missing in VAL"
assert set(y_test.unique()) == {0,1,2}, "Class missing in TEST"

X_test.to_pickle("outputs/X_test.pkl")
y_test.to_pickle("outputs/y_test.pkl")

print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

# 5. Class imbalance handling
weights = compute_sample_weight(class_weight="balanced", y=y_train)
weights[y_train == 2] *= 10
sample_weights = weights

# 6. DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

# 7. Model parameters
num_class = 3

params = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "eta": 0.05,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": RANDOM_STATE
}

# 8. Train
watchlist = [(dtrain, "train"), (dval, "val")]
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=500,
    evals=watchlist,
    early_stopping_rounds=50,
    verbose_eval=50
)

# 9. Save model + features
bst.save_model(MODEL_OUT)
joblib.dump(features, FEATURES_OUT)
print("Model saved:", MODEL_OUT)

# 10. Evaluation
y_pred_proba = bst.predict(dtest)

t1 = 0.4
t2 = 0.12

y_pred = []
for p in y_pred_proba:
    if p[2] >= t2:
        y_pred.append(2)
    elif p[1] >= t1:
        y_pred.append(1)
    else:
        y_pred.append(0)

y_pred = np.array(y_pred)

print("\nClassification Report")
print(classification_report(y_test, y_pred))
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

# 11. AUC
y_test_bin = np.zeros((len(y_test), num_class))
for i, label in enumerate(y_test):
    y_test_bin[i, label] = 1

aucs = []
for c in range(num_class):
    fpr, tpr, _ = roc_curve(y_test_bin[:, c], y_pred_proba[:, c])
    aucs.append(auc(fpr, tpr))

print("Per-class AUC:", aucs)
print("Macro AUC:", np.mean(aucs))

# 12. SHAP
explainer = shap.TreeExplainer(bst)
X_shap = X_test.sample(min(200, len(X_test)), random_state=RANDOM_STATE)
shap_values = explainer.shap_values(X_shap)

shap.summary_plot(shap_values, X_shap, show=False)
plt.tight_layout()
plt.savefig(SHAP_PLOT_OUT, dpi=200)
plt.close()

print("SHAP plot saved:", SHAP_PLOT_OUT)

# 13. Save predictions
out_df = X_test.copy()
out_df["y_true"] = y_test.values
out_df["y_pred"] = y_pred

for c in range(num_class):
    out_df[f"proba_class_{c}"] = y_pred_proba[:, c]

out_df.to_csv(PRED_OUT, index=False)
print("Predictions saved:", PRED_OUT)

print("\nTraining pipeline completed successfully.")
