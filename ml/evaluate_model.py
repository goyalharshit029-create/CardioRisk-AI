from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

import matplotlib.pyplot as plt


ROOT = Path(__file__).parent

df = pd.read_csv(ROOT / "cleaned_cardio.csv")

model = joblib.load(
    ROOT.parent / "backend" / "cardiovascular_model.pkl"
)

features = joblib.load(
    ROOT.parent / "backend" / "features.pkl"
)


X = df[features]
y = df["cardio"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


pred = model.predict(X_test)

prob = model.predict_proba(X_test)[:, 1]


accuracy = accuracy_score(y_test, pred)
precision = precision_score(y_test, pred)
recall = recall_score(y_test, pred)
f1 = f1_score(y_test, pred)
auc = roc_auc_score(y_test, prob)


print("\n===== CARDIORISK AI MODEL EVALUATION =====")

print(f"Accuracy : {accuracy:.6f}")
print(f"Precision: {precision:.6f}")
print(f"Recall   : {recall:.6f}")
print(f"F1 Score : {f1:.6f}")
print(f"ROC-AUC  : {auc:.6f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))


# ROC Curve

fpr, tpr, _ = roc_curve(y_test, prob)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"Gradient Boosting AUC = {auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    "--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("CardioRisk AI - ROC Curve")

plt.legend()

plt.tight_layout()

plt.show()