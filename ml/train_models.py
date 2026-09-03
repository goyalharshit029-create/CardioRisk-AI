from pathlib import Path
import json
import warnings
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV
)

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).parent
CSV = ROOT / "cleaned_cardio.csv"
BACKEND = ROOT.parent / "backend"

BACKEND.mkdir(exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV)

print("\nDataset shape:", df.shape)
print("Columns:", df.columns.tolist())


# ============================================================
# BASIC CLEANING
# ============================================================

if "id" in df.columns:
    df = df.drop(columns=["id"])


# Blood pressure cleaning
if "ap_hi" in df.columns and "ap_lo" in df.columns:

    df = df[
        (df["ap_hi"] >= 80) &
        (df["ap_hi"] <= 250) &
        (df["ap_lo"] >= 40) &
        (df["ap_lo"] <= 150) &
        (df["ap_hi"] > df["ap_lo"])
    ].copy()


# Height cleaning
if "height" in df.columns:
    df = df[
        (df["height"] >= 140) &
        (df["height"] <= 210)
    ].copy()


# Weight cleaning
if "weight" in df.columns:
    df = df[
        (df["weight"] >= 35) &
        (df["weight"] <= 200)
    ].copy()


print("Dataset after cleaning:", df.shape)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

# ------------------------------------------------------------
# BMI
# ------------------------------------------------------------

if "height" in df.columns and "weight" in df.columns:

    height_m = df["height"] / 100

    df["bmi"] = df["weight"] / (height_m ** 2)


# ------------------------------------------------------------
# AGE
# ------------------------------------------------------------

if "age_years" not in df.columns and "age" in df.columns:
    df["age_years"] = df["age"] / 365.25


if "age_years" in df.columns:

    df["age_squared"] = df["age_years"] ** 2

    df["age_cubed"] = df["age_years"] ** 3

    df["age_over_40"] = (df["age_years"] >= 40).astype(int)
    df["age_over_50"] = (df["age_years"] >= 50).astype(int)
    df["age_over_60"] = (df["age_years"] >= 60).astype(int)


# ------------------------------------------------------------
# BLOOD PRESSURE
# ------------------------------------------------------------

if "ap_hi" in df.columns and "ap_lo" in df.columns:

    df["pulse_pressure"] = (
        df["ap_hi"] - df["ap_lo"]
    )

    df["mean_arterial_pressure"] = (
        df["ap_hi"] + 2 * df["ap_lo"]
    ) / 3

    df["bp_ratio"] = (
        df["ap_hi"] /
        df["ap_lo"].replace(0, np.nan)
    )

    df["high_bp"] = (
        (df["ap_hi"] >= 140) |
        (df["ap_lo"] >= 90)
    ).astype(int)

    df["very_high_bp"] = (
        (df["ap_hi"] >= 160) |
        (df["ap_lo"] >= 100)
    ).astype(int)

    df["normal_bp"] = (
        (df["ap_hi"] < 120) &
        (df["ap_lo"] < 80)
    ).astype(int)

    df["bp_sum"] = (
        df["ap_hi"] + df["ap_lo"]
    )

    df["bp_squared"] = (
        df["ap_hi"] ** 2
    )


# ------------------------------------------------------------
# BMI FEATURES
# ------------------------------------------------------------

if "bmi" in df.columns:

    df["bmi_squared"] = df["bmi"] ** 2

    df["underweight"] = (
        df["bmi"] < 18.5
    ).astype(int)

    df["overweight"] = (
        df["bmi"] >= 25
    ).astype(int)

    df["obese"] = (
        df["bmi"] >= 30
    ).astype(int)

    df["severely_obese"] = (
        df["bmi"] >= 35
    ).astype(int)


# ------------------------------------------------------------
# CHOLESTEROL
# ------------------------------------------------------------

if "cholesterol" in df.columns:

    df["high_cholesterol"] = (
        df["cholesterol"] >= 2
    ).astype(int)

    df["very_high_cholesterol"] = (
        df["cholesterol"] == 3
    ).astype(int)


# ------------------------------------------------------------
# GLUCOSE
# ------------------------------------------------------------

if "gluc" in df.columns:

    df["high_glucose"] = (
        df["gluc"] >= 2
    ).astype(int)

    df["very_high_glucose"] = (
        df["gluc"] == 3
    ).astype(int)


# ------------------------------------------------------------
# LIFESTYLE RISK SCORE
# ------------------------------------------------------------

risk_columns = []

if "smoke" in df.columns:
    risk_columns.append("smoke")

if "alco" in df.columns:
    risk_columns.append("alco")

if "active" in df.columns:
    df["inactive"] = 1 - df["active"]
    risk_columns.append("inactive")


if risk_columns:
    df["lifestyle_risk_score"] = df[
        risk_columns
    ].sum(axis=1)


# ------------------------------------------------------------
# INTERACTIONS
# ------------------------------------------------------------

if "age_years" in df.columns and "ap_hi" in df.columns:
    df["age_bp_interaction"] = (
        df["age_years"] * df["ap_hi"]
    )


if "age_years" in df.columns and "ap_lo" in df.columns:
    df["age_diastolic_interaction"] = (
        df["age_years"] * df["ap_lo"]
    )


if "age_years" in df.columns and "bmi" in df.columns:
    df["age_bmi_interaction"] = (
        df["age_years"] * df["bmi"]
    )


if "bmi" in df.columns and "ap_hi" in df.columns:
    df["bmi_bp_interaction"] = (
        df["bmi"] * df["ap_hi"]
    )


if "cholesterol" in df.columns and "age_years" in df.columns:
    df["cholesterol_age"] = (
        df["cholesterol"] * df["age_years"]
    )


if "gluc" in df.columns and "bmi" in df.columns:
    df["glucose_bmi"] = (
        df["gluc"] * df["bmi"]
    )


# ------------------------------------------------------------
# COMBINED CLINICAL RISK
# ------------------------------------------------------------

clinical_risk = pd.Series(
    0,
    index=df.index
)

if "high_bp" in df.columns:
    clinical_risk += df["high_bp"]

if "high_cholesterol" in df.columns:
    clinical_risk += df["high_cholesterol"]

if "high_glucose" in df.columns:
    clinical_risk += df["high_glucose"]

if "obese" in df.columns:
    clinical_risk += df["obese"]

if "age_over_50" in df.columns:
    clinical_risk += df["age_over_50"]

df["clinical_risk_score"] = clinical_risk


# Replace infinity
df = df.replace([np.inf, -np.inf], np.nan)


print("\nFeature engineering complete.")
print("Final dataset shape:", df.shape)


# ============================================================
# FEATURES / TARGET
# ============================================================

features = [
    col for col in df.columns
    if col != "cardio"
]

X = df[features]
y = df["cardio"]

print("Total features:", len(features))


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


# ============================================================
# PREPROCESSING
# ============================================================

preprocess_scaled = Pipeline([

    (
        "imputer",
        SimpleImputer(strategy="median")
    ),

    (
        "scaler",
        StandardScaler()
    )

])


preprocess_tree = Pipeline([

    (
        "imputer",
        SimpleImputer(strategy="median")
    )

])


# ============================================================
# CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=42
)


# ============================================================
# MODELS
# ============================================================

models = {

    "Logistic Regression": (

        LogisticRegression(
            max_iter=5000,
            random_state=42
        ),

        preprocess_scaled,

        {
            "model__C": [
                0.01,
                0.05,
                0.1,
                0.5,
                1,
                5,
                10,
                20
            ]
        }
    ),


    "Random Forest": (

        RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        ),

        preprocess_tree,

        {
            "model__n_estimators": [
                300,
                500
            ],

            "model__max_depth": [
                8,
                12,
                16,
                20,
                None
            ],

            "model__min_samples_split": [
                2,
                5,
                10
            ],

            "model__min_samples_leaf": [
                1,
                2,
                4
            ],

            "model__max_features": [
                "sqrt",
                "log2"
            ]
        }
    ),


    "Extra Trees": (

        ExtraTreesClassifier(
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        ),

        preprocess_tree,

        {
            "model__n_estimators": [
                300,
                500
            ],

            "model__max_depth": [
                10,
                15,
                20,
                None
            ],

            "model__min_samples_split": [
                2,
                5,
                10
            ],

            "model__min_samples_leaf": [
                1,
                2,
                4
            ],

            "model__max_features": [
                "sqrt",
                "log2"
            ]
        }
    ),


    "Gradient Boosting": (

        GradientBoostingClassifier(
            random_state=42
        ),

        preprocess_tree,

        {
            "model__n_estimators": [
                150,
                250,
                350
            ],

            "model__learning_rate": [
                0.02,
                0.03,
                0.05,
                0.08
            ],

            "model__max_depth": [
                2,
                3,
                4
            ],

            "model__min_samples_split": [
                2,
                5,
                10
            ],

            "model__min_samples_leaf": [
                2,
                5,
                10
            ],

            "model__subsample": [
                0.7,
                0.8,
                0.9,
                1.0
            ]
        }
    ),


    "Hist Gradient Boosting": (

        HistGradientBoostingClassifier(
            random_state=42,
            early_stopping=True
        ),

        preprocess_tree,

        {
            "model__learning_rate": [
                0.02,
                0.03,
                0.05,
                0.08,
                0.1
            ],

            "model__max_iter": [
                200,
                300,
                500
            ],

            "model__max_leaf_nodes": [
                15,
                31,
                63
            ],

            "model__min_samples_leaf": [
                10,
                20,
                30
            ],

            "model__l2_regularization": [
                0.0,
                0.1,
                0.5,
                1.0
            ]
        }
    )

}


# ============================================================
# TRY XGBOOST
# ============================================================

try:

    from xgboost import XGBClassifier

    print("\n✅ XGBoost detected!")

    models["XGBoost"] = (

        XGBClassifier(

            random_state=42,

            eval_metric="logloss",

            n_jobs=-1
        ),

        preprocess_tree,

        {

            "model__n_estimators": [
                300,
                500,
                700
            ],

            "model__max_depth": [
                3,
                4,
                5,
                6
            ],

            "model__learning_rate": [
                0.02,
                0.03,
                0.05
            ],

            "model__subsample": [
                0.7,
                0.8,
                0.9
            ],

            "model__colsample_bytree": [
                0.7,
                0.8,
                1.0
            ],

            "model__min_child_weight": [
                1,
                3,
                5
            ]

        }
    )

except ImportError:

    print(
        "\n⚠️ XGBoost not installed."
    )

    print(
        "To install: pip install xgboost"
    )


# ============================================================
# TRAIN MODELS
# ============================================================

results = []

trained_models = {}

best_auc = -1

best_model = None

best_name = None


for name, (model, preprocess, params) in models.items():

    print("\n")

    print("=" * 60)

    print("Training:", name)

    print("=" * 60)


    pipeline = Pipeline([

        ("preprocess", preprocess),

        ("model", model)

    ])


    search = RandomizedSearchCV(

        estimator=pipeline,

        param_distributions=params,

        n_iter=20,

        scoring="roc_auc",

        cv=cv,

        n_jobs=-1,

        random_state=42,

        verbose=1

    )


    search.fit(
        X_train,
        y_train
    )


    trained_model = search.best_estimator_

    trained_models[name] = trained_model


    print(
        "\nBest CV ROC-AUC:",
        round(search.best_score_, 6)
    )

    print(
        "Best Parameters:",
        search.best_params_
    )


    predictions = trained_model.predict(X_test)

    probabilities = trained_model.predict_proba(X_test)[:, 1]


    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        probabilities
    )


    results.append({

        "model": name,

        "accuracy": float(accuracy),

        "precision": float(precision),

        "recall": float(recall),

        "f1": float(f1),

        "roc_auc": float(auc)

    })


    print(
        "\nTest ROC-AUC:",
        round(auc, 6)
    )


    if auc > best_auc:

        best_auc = auc

        best_model = trained_model

        best_name = name


# ============================================================
# ENSEMBLE OF TOP MODELS
# ============================================================

print("\n")

print("=" * 60)

print("TRYING PROBABILITY ENSEMBLE")

print("=" * 60)


model_scores = sorted(
    results,
    key=lambda x: x["roc_auc"],
    reverse=True
)


top_names = [
    item["model"]
    for item in model_scores[:3]
]


if len(top_names) >= 2:

    print("Top models:", top_names)


    probs = []

    for name in top_names:

        prob = trained_models[name].predict_proba(
            X_test
        )[:, 1]

        probs.append(prob)


    # Average probabilities
    ensemble_prob = np.mean(
        probs,
        axis=0
    )


    ensemble_pred = (
        ensemble_prob >= 0.5
    ).astype(int)


    ensemble_auc = roc_auc_score(
        y_test,
        ensemble_prob
    )


    ensemble_accuracy = accuracy_score(
        y_test,
        ensemble_pred
    )


    ensemble_precision = precision_score(
        y_test,
        ensemble_pred,
        zero_division=0
    )


    ensemble_recall = recall_score(
        y_test,
        ensemble_pred,
        zero_division=0
    )


    ensemble_f1 = f1_score(
        y_test,
        ensemble_pred,
        zero_division=0
    )


    print(
        "\nEnsemble ROC-AUC:",
        round(ensemble_auc, 6)
    )


    results.append({

        "model": "Top Model Probability Ensemble",

        "accuracy": float(ensemble_accuracy),

        "precision": float(ensemble_precision),

        "recall": float(ensemble_recall),

        "f1": float(ensemble_f1),

        "roc_auc": float(ensemble_auc)

    })


# ============================================================
# SAVE BEST INDIVIDUAL MODEL
# ============================================================

# Note:
# We save the best actual trained pipeline.
# The probability ensemble is only reported unless
# it beats the individual model and needs custom deployment.

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "roc_auc",
    ascending=False
)


best_row = results_df.iloc[0].to_dict()


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    best_model,
    BACKEND / "cardiovascular_model.pkl"
)


joblib.dump(
    features,
    BACKEND / "features.pkl"
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {

    "selected_model": best_name,

    "accuracy": float(best_row["accuracy"]),

    "precision": float(best_row["precision"]),

    "recall": float(best_row["recall"]),

    "f1": float(best_row["f1"]),

    "roc_auc": float(best_row["roc_auc"]),

    "models": results_df.to_dict(
        orient="records"
    ),

    "dataset_records": int(len(df)),

    "feature_count": int(len(features)),

    "split":
        "80/20 stratified train-test split with 5-fold cross-validation tuning"

}


(BACKEND / "metrics.json").write_text(

    json.dumps(
        metrics,
        indent=2
    )

)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n\n")

print("=" * 70)

print("FINAL RESULTS")

print("=" * 70)


print(
    results_df.to_string(
        index=False
    )
)


print("\n🏆 SELECTED MODEL:", best_name)

print(
    "🏆 BEST ROC-AUC:",
    round(best_auc, 6)
)


print(
    "\nTotal Features Used:",
    len(features)
)


print("\n✅ Model saved successfully!")

print(
    "\nSaved model:",
    BACKEND / "cardiovascular_model.pkl"
)

print(
    "Saved metrics:",
    BACKEND / "metrics.json"
)