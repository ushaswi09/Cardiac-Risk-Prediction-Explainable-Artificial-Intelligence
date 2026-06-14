import os
import pandas as pd
import joblib
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

os.chdir("/Users/ushaswikaruturi/Desktop/project")

OUT_DIR = "outputs"
X_test = pd.read_pickle(os.path.join(OUT_DIR, "X_test.pkl"))
features = joblib.load(os.path.join(OUT_DIR, "xgb_feature_list.pkl"))

bst = xgb.Booster()
bst.load_model(os.path.join(OUT_DIR, "xgb_model.json"))

X_shap = X_test.sample(min(100, len(X_test)), random_state=42)

explainer = shap.TreeExplainer(bst)
shap_values = explainer.shap_values(X_shap)

print("shap_values type:", type(shap_values))
if isinstance(shap_values, list):
    print("num classes:", len(shap_values))
else:
    print("shap_values shape:", shap_values.shape)

# Handle both list-of-arrays (older shap) and 3D array (newer shap) formats
if isinstance(shap_values, list):
    class_arrays = shap_values
else:
    # shape (n_samples, n_features, n_classes)
    class_arrays = [shap_values[:, :, c] for c in range(shap_values.shape[2])]

for c, sv in enumerate(class_arrays):
    plt.figure()
    shap.summary_plot(sv, X_shap, show=False)
    plt.title(f"SHAP Summary - Class {c}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"shap_class_{c}.png"), dpi=200)
    plt.close()
    print(f"Saved shap_class_{c}.png")

# Global importance (mean abs across classes)
import numpy as np
mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in class_arrays], axis=0)
order = np.argsort(mean_abs)[::-1]

plt.figure(figsize=(8,10))
plt.barh(range(len(features)), mean_abs[order][::-1], color="skyblue", edgecolor="black")
plt.yticks(range(len(features)), [features[i] for i in order][::-1])
plt.xlabel("Mean |SHAP value| (Average Impact on Model Output)")
plt.title("Global SHAP Feature Importance (Aggregated across Classes)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "shap_global_importance.png"), dpi=200)
plt.close()
print("Saved shap_global_importance.png")
