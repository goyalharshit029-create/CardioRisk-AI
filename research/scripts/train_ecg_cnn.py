from pathlib import Path
import ast
import numpy as np
import pandas as pd
import wfdb
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, RocCurveDisplay

import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

ECG_DIR = ROOT / "research" / "datasets" / "ecg"

MODEL_DIR = ROOT / "research" / "models"
RESULT_DIR = ROOT / "research" / "experiments"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 65)
print("CARDIORISK AI - ECG CNN MODEL")
print("=" * 65)


# ============================================================
# CONFIGURATION
# ============================================================

MAX_RECORDS = 2000
SIGNAL_LENGTH = 1000
RANDOM_STATE = 42


# ============================================================
# FIND PTB-XL DATABASE
# ============================================================

csv_files = list(ECG_DIR.rglob("ptbxl_database.csv"))

if not csv_files:
    raise FileNotFoundError(
        "\nCould not find ptbxl_database.csv\n"
        "Make sure the PTB-XL dataset is extracted inside:\n"
        f"{ECG_DIR}"
    )

DATABASE_PATH = csv_files[0]

DATASET_ROOT = DATABASE_PATH.parent

print("\nDataset found:")
print(DATASET_ROOT)


# ============================================================
# LOAD METADATA
# ============================================================

df = pd.read_csv(
    DATABASE_PATH,
    index_col="ecg_id"
)

print("\nTotal ECG records:", len(df))


# ============================================================
# CREATE BINARY LABEL
# ============================================================

def create_label(row):
    """
    Binary classification:

    1 = abnormal ECG
    0 = normal ECG
    """

    statements = ast.literal_eval(
        row["scp_codes"]
    )

    if "NORM" in statements:
        return 0

    return 1


df["label"] = df.apply(
    create_label,
    axis=1
)

print("\nLabel distribution:")
print(df["label"].value_counts())


# ============================================================
# SAMPLE DATA
# ============================================================

normal = df[df["label"] == 0]

abnormal = df[df["label"] == 1]

samples_per_class = MAX_RECORDS // 2

normal_sample = normal.sample(
    n=min(samples_per_class, len(normal)),
    random_state=RANDOM_STATE
)

abnormal_sample = abnormal.sample(
    n=min(samples_per_class, len(abnormal)),
    random_state=RANDOM_STATE
)

df_sample = pd.concat(
    [
        normal_sample,
        abnormal_sample
    ]
).sample(
    frac=1,
    random_state=RANDOM_STATE
)

print("\nSelected ECG records:", len(df_sample))


# ============================================================
# LOAD ECG SIGNALS
# ============================================================

signals = []
labels = []

print("\nLoading ECG signals...")

for i, (_, row) in enumerate(
    df_sample.iterrows()
):

    try:

        record_path = DATASET_ROOT / row["filename_lr"]

        signal, metadata = wfdb.rdsamp(
            str(record_path)
        )

        signal = signal[:SIGNAL_LENGTH]

        # Skip invalid signals
        if signal.shape[0] < SIGNAL_LENGTH:
            continue

        signals.append(signal)

        labels.append(row["label"])

        if len(signals) % 100 == 0:

            print(
                f"Loaded {len(signals)} ECG records"
            )

    except Exception as e:

        print(
            f"Skipping record {i}:",
            e
        )


X = np.array(
    signals,
    dtype=np.float32
)

y = np.array(
    labels,
    dtype=np.int32
)


print("\nFinal ECG dataset shape:")
print("X:", X.shape)
print("y:", y.shape)


# ============================================================
# NORMALIZATION
# ============================================================

print("\nNormalizing ECG signals...")

mean = X.mean(
    axis=(1, 2),
    keepdims=True
)

std = X.std(
    axis=(1, 2),
    keepdims=True
)

X = (X - mean) / (
    std + 1e-8
)


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=RANDOM_STATE,

    stratify=y

)


print("\nTraining ECG samples:", len(X_train))

print("Testing ECG samples:", len(X_test))


# ============================================================
# CNN MODEL
# ============================================================

print("\nBuilding CNN model...")


model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=X_train.shape[1:]
    ),

    tf.keras.layers.Conv1D(
        32,
        kernel_size=7,
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling1D(
        pool_size=2
    ),


    tf.keras.layers.Conv1D(
        64,
        kernel_size=5,
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling1D(
        pool_size=2
    ),


    tf.keras.layers.Conv1D(
        128,
        kernel_size=3,
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.GlobalAveragePooling1D(),


    tf.keras.layers.Dropout(
        0.4
    ),


    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),


    tf.keras.layers.Dropout(
        0.3
    ),


    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )

])


model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(
            name="auc"
        )
    ]

)


model.summary()


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    tf.keras.callbacks.EarlyStopping(

        monitor="val_auc",

        mode="max",

        patience=5,

        restore_best_weights=True

    )

]


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining ECG CNN...")


history = model.fit(

    X_train,
    y_train,

    validation_split=0.2,

    epochs=30,

    batch_size=32,

    callbacks=callbacks,

    verbose=1

)


# ============================================================
# EVALUATE
# ============================================================

print("\nEvaluating model...")


results = model.evaluate(

    X_test,

    y_test,

    verbose=0

)


y_prob = model.predict(
    X_test,
    verbose=0
).flatten()


y_pred = (
    y_prob >= 0.5
).astype(int)


roc_auc = roc_auc_score(
    y_test,
    y_prob
)


print("\n" + "=" * 65)

print("ECG CNN RESULTS")

print("=" * 65)


print(
    f"\nTest Accuracy: "
    f"{results[1]:.4f}"
)

print(
    f"Test AUC: "
    f"{roc_auc:.4f}"
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = MODEL_DIR / "ecg_cnn.keras"

model.save(
    model_path
)

print("\nModel saved:")

print(model_path)


# ============================================================
# ROC CURVE
# ============================================================

plt.figure(
    figsize=(8, 6)
)

RocCurveDisplay.from_predictions(
    y_test,
    y_prob
)

plt.title(
    f"ECG CNN ROC Curve "
    f"(AUC = {roc_auc:.3f})"
)

plt.tight_layout()


roc_path = (
    RESULT_DIR /
    "ecg_cnn_roc_curve.png"
)

plt.savefig(
    roc_path,
    dpi=150
)

plt.close()


print("\nROC curve saved:")

print(roc_path)


# ============================================================
# SAVE RESULTS
# ============================================================

import json


metrics = {

    "test_accuracy":
        float(results[1]),

    "roc_auc":
        float(roc_auc),

    "total_samples":
        int(len(X)),

    "signal_length":
        SIGNAL_LENGTH

}


metrics_path = (
    RESULT_DIR /
    "ecg_cnn_metrics.json"
)


with open(
    metrics_path,
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


print("\nMetrics saved:")

print(metrics_path)


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 65)

print("STEP 5 COMPLETE!")

print("=" * 65)