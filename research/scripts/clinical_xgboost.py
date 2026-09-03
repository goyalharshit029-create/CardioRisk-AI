import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
    RocCurveDisplay
)

from xgboost import XGBClassifier
import shap
import joblib


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "research" / "datasets" / "clinical"
MODEL_DIR = ROOT / "research" / "models"
RESULT_DIR = ROOT / "research" / "experiments"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 65)
print("CARDIORISK AI - CLINICAL XGBOOST MODEL")
print("=" * 65)


# ============================================================
# CLEVELAND DATASET
# ============================================================

COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target"
]

file_path = DATA_DIR / "processed.cleveland.data"

if not file_path.exists():
    raise FileNotFoundError(
        f"\nDataset not found:\n{file_path}\n"
    )


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    file_path,
    names=COLUMNS
)

print("\nDataset loaded successfully!")
print("Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# CLEAN DATA
# ============================================================

print("\nCleaning dataset...")

df = df.replace("?", np.nan)

for col in COLUMNS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("\nMissing values:")
print(df.isnull().sum())

# Fill missing values using median
for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())


# ============================================================
# TARGET CONVERSION
# ============================================================

# Original target:
# 0 = no heart disease
# 1,2,3,4 = heart disease

df["target"] = (df["target"] > 0).astype(int)

print("\nTarget distribution:")
print(df["target"].value_counts())


# ============================================================
# FEATURE ENGINEERING
# ============================================================

print("\nPerforming feature engineering...")

# Age categories
df["age_risk"] = pd.cut(
    df["age"],
    bins=[0, 40, 50, 60, 70, 100],
    labels=False
)

# Blood pressure category
df["bp_risk"] = pd.cut(
    df["trestbps"],
    bins=[0, 120, 130, 140, 180, 300],
    labels=False
)

# Cholesterol risk
df["chol_risk"] = pd.cut(
    df["chol"],
    bins=[0, 200, 240, 300, 1000],
    labels=False
)

# Interaction features
df["age_chol"] = df["age"] * df["chol"]
df["age_bp"] = df["age"] * df["trestbps"]

# Heart rate reserve style feature
df["age_hr_ratio"] = df["thalach"] / (220 - df["age"])

# ST depression risk
df["st_risk"] = df["oldpeak"] * df["exang"]


# ============================================================
# FEATURES / TARGET
# ============================================================

X = df.drop("target", axis=1)
y = df["target"]

print("\nTotal features:", X.shape[1])
print("Features:")
print(list(X.columns))


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# XGBOOST MODEL
# ============================================================

print("\nTraining XGBoost...")

model = XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=2,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)


# ============================================================
# PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

print("\n" + "=" * 65)
print("MODEL RESULTS")
print("=" * 65)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = MODEL_DIR / "clinical_xgboost.pkl"

joblib.dump(
    {
        "model": model,
        "features": list(X.columns)
    },
    model_path
)

print("\nModel saved:")
print(model_path)


# ============================================================
# ROC CURVE
# ============================================================

plt.figure(figsize=(8, 6))

RocCurveDisplay.from_predictions(
    y_test,
    y_prob
)

plt.title(
    f"Clinical XGBoost ROC Curve (AUC = {roc_auc:.3f})"
)

plt.tight_layout()

roc_path = RESULT_DIR / "clinical_roc_curve.png"

plt.savefig(
    roc_path,
    dpi=150
)

plt.close()

print("\nROC curve saved:")
print(roc_path)


# ============================================================
# SHAP EXPLAINABILITY
# ============================================================

print("\nGenerating SHAP explanations...")

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test)


# ------------------------------------------------------------
# SHAP FEATURE IMPORTANCE
# ------------------------------------------------------------

plt.figure()

shap.summary_plot(
    shap_values,
    X_test,
    plot_type="bar",
    show=False
)

plt.tight_layout()

shap_bar_path = RESULT_DIR / "clinical_shap_importance.png"

plt.savefig(
    shap_bar_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# SHAP SUMMARY PLOT
# ------------------------------------------------------------

plt.figure()

shap.summary_plot(
    shap_values,
    X_test,
    show=False
)

plt.tight_layout()

shap_summary_path = RESULT_DIR / "clinical_shap_summary.png"

plt.savefig(
    shap_summary_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


print("\nSHAP plots saved:")
print(shap_bar_path)
print(shap_summary_path)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {
    "accuracy": float(accuracy),
    "roc_auc": float(roc_auc),
    "training_samples": int(len(X_train)),
    "testing_samples": int(len(X_test)),
    "total_features": int(X.shape[1])
}

metrics_path = RESULT_DIR / "clinical_metrics.json"

import json

with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)


print("\nMetrics saved:")
print(metrics_path)


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 65)
print("STEP 4 COMPLETE!")
print("=" * 65)

print("\nFinal ROC-AUC:", round(roc_auc, 4))
print("Clinical XGBoost + SHAP completed successfully.")