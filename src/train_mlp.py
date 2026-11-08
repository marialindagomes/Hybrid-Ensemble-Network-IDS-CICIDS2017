# ============================================================
# STEP 10B - LIGHTWEIGHT MLP / DNN BASELINE
# CICIDS2017 Hybrid ML IDS
# ============================================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import tensorflow as tf  # type: ignore[reportMissingModuleSource]
import os
import time
import warnings

import numpy as np
import pandas as pd

EarlyStopping = tf.keras.callbacks.EarlyStopping
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
Input = tf.keras.layers.Input
Sequential = tf.keras.Sequential

warnings.filterwarnings("ignore")

# ============================================================
# TENSORFLOW
# ============================================================


# ============================================================
# SKLEARN
# ============================================================


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROCESSED_DIR = os.path.join(
    BASE_DIR, "data", "processed"
)

RESULTS_DIR = os.path.join(
    BASE_DIR, "results"
)

MODELS_DIR = os.path.join(
    BASE_DIR, "models"
)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ============================================================
# STEP HEADER
# ============================================================

print("=" * 70)
print("STEP 10B - LIGHTWEIGHT MLP / DNN BASELINE")
print("=" * 70)

# ============================================================
# CHECK REQUIRED FILES
# ============================================================

X_DL_PATH = os.path.join(
    PROCESSED_DIR,
    "X_dl_train.csv"
)

Y_DL_PATH = os.path.join(
    PROCESSED_DIR,
    "y_dl_train.csv"
)

X_TEST_PATH = os.path.join(
    PROCESSED_DIR,
    "X_test_scaled.csv"
)

Y_TEST_PATH = os.path.join(
    PROCESSED_DIR,
    "y_test.csv"
)

LABEL_ENCODER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "label_encoder.pkl"
)

required_files = [
    X_DL_PATH,
    Y_DL_PATH,
    X_TEST_PATH,
    Y_TEST_PATH
]

print("\nChecking required files...")

for file in required_files:

    if not os.path.exists(file):

        print("\nERROR: Required file not found:")
        print(file)

        raise FileNotFoundError(file)

print("Required files found successfully.")

# ============================================================
# LOAD DL TRAINING DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING LIGHTWEIGHT DL TRAINING DATA")
print("=" * 70)

X_train = pd.read_csv(X_DL_PATH).values.astype(np.float32)

y_train = pd.read_csv(Y_DL_PATH).values.ravel()

print("\nX_train shape:")
print(X_train.shape)

print("\ny_train shape:")
print(y_train.shape)

# ============================================================
# LOAD TEST DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING ORIGINAL TEST DATA")
print("=" * 70)

X_test = pd.read_csv(X_TEST_PATH).values.astype(np.float32)

y_test = pd.read_csv(Y_TEST_PATH).values.ravel()

print("\nX_test shape:")
print(X_test.shape)

print("\ny_test shape:")
print(y_test.shape)

# ============================================================
# VALIDATION
# ============================================================

if X_train.shape[0] != y_train.shape[0]:

    raise ValueError(
        "Training feature-label sample count mismatch."
    )

if X_test.shape[0] != y_test.shape[0]:

    raise ValueError(
        "Test feature-label sample count mismatch."
    )

print("\nFeature-label consistency check passed.")

# ============================================================
# BASIC INFORMATION
# ============================================================

num_features = X_train.shape[1]

num_classes = len(np.unique(y_train))

print("\nNumber of features:", num_features)
print("Number of classes :", num_classes)

print("\nTraining samples:", len(X_train))
print("Test samples    :", len(X_test))

# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("DL TRAINING CLASS DISTRIBUTION")
print("=" * 70)

unique, counts = np.unique(
    y_train,
    return_counts=True
)

for cls, count in zip(unique, counts):

    print(
        f"Class {cls:2d} : {count:,}"
    )

# ============================================================
# CREATE LIGHTWEIGHT MLP
# ============================================================

print("\n" + "=" * 70)
print("CREATING LIGHTWEIGHT MLP")
print("=" * 70)

print("""
Architecture:

Input: 30 features
        |
Dense: 64 neurons
        |
ReLU
        |
Dropout: 0.20
        |
Dense: 32 neurons
        |
ReLU
        |
Dense: 15 neurons
        |
Softmax
""")

model = Sequential([

    Input(shape=(num_features,)),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(0.20),

    Dense(
        32,
        activation="relu"
    ),

    Dense(
        num_classes,
        activation="softmax"
    )

])

# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)

print("Model created successfully.")

print("\nModel summary:")

model.summary()

# ============================================================
# COUNT PARAMETERS
# ============================================================

total_parameters = model.count_params()

print("\nTotal trainable parameters:")
print(f"{total_parameters:,}")

# ============================================================
# EARLY STOPPING
# ============================================================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=3,

    restore_best_weights=True

)

# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 70)
print("TRAINING LIGHTWEIGHT MLP")
print("=" * 70)

print("\nTraining started...")

training_start = time.perf_counter()

history = model.fit(

    X_train,
    y_train,

    validation_split=0.10,

    epochs=15,

    batch_size=256,

    callbacks=[early_stopping],

    verbose=1

)

training_end = time.perf_counter()

training_time = (
    training_end -
    training_start
)

print("\nTraining completed.")

print(
    f"Training time: {training_time:.4f} seconds"
)

print(
    f"Training time: {training_time / 60:.2f} minutes"
)

# ============================================================
# SAVE MLP MODEL
# ============================================================

MLP_MODEL_PATH = os.path.join(

    MODELS_DIR,

    "lightweight_mlp_ids.keras"

)

model.save(MLP_MODEL_PATH)

print("\nModel saved:")

print(MLP_MODEL_PATH)

# ============================================================
# INFERENCE
# ============================================================

print("\n" + "=" * 70)
print("MLP INFERENCE")
print("=" * 70)

print("\nGenerating predictions...")

# Warm-up prediction
model.predict(
    X_test[:100],
    batch_size=256,
    verbose=0
)

inference_start = time.perf_counter()

probabilities = model.predict(

    X_test,

    batch_size=256,

    verbose=0

)

inference_end = time.perf_counter()

inference_time = (
    inference_end -
    inference_start
)

# Convert probabilities to class predictions

pred = np.argmax(
    probabilities,
    axis=1
)

# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    pred
)

precision = precision_score(
    y_test,
    pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    pred,
    average="weighted",
    zero_division=0
)

latency_ms = (
    inference_time /
    len(X_test)
) * 1000

# ============================================================
# MODEL SIZE
# ============================================================

model_size_bytes = os.path.getsize(
    MLP_MODEL_PATH
)

model_size_mb = (
    model_size_bytes /
    (1024 ** 2)
)

# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("MLP BASELINE RESULTS")
print("=" * 70)

print(
    f"Accuracy        : {accuracy:.6f}"
)

print(
    f"Precision       : {precision:.6f}"
)

print(
    f"Recall          : {recall:.6f}"
)

print(
    f"F1 Score        : {f1:.6f}"
)

print(
    f"Training Time   : {training_time:.4f} seconds"
)

print(
    f"Training Time   : {training_time / 60:.2f} minutes"
)

print(
    f"Inference Time  : {inference_time:.4f} seconds"
)

print(
    f"Latency/sample  : {latency_ms:.6f} ms"
)

print(
    f"Model Size      : {model_size_mb:.2f} MB"
)

print(
    f"Parameters      : {total_parameters:,}"
)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(

    y_test,
    pred,
    zero_division=0

)

print(report)

report_path = os.path.join(

    RESULTS_DIR,

    "mlp_classification_report.txt"

)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(report)

print(
    "\nClassification report saved:"
)

print(report_path)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    pred
)

cm_path = os.path.join(

    RESULTS_DIR,

    "mlp_confusion_matrix.csv"

)

pd.DataFrame(cm).to_csv(
    cm_path,
    index=False
)

print("\nConfusion matrix saved:")

print(cm_path)

# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(
    history.history
)

history_path = os.path.join(

    RESULTS_DIR,

    "mlp_training_history.csv"

)

history_df.to_csv(
    history_path,
    index=False
)

print("\nTraining history saved:")

print(history_path)

# ============================================================
# SAVE COMPUTATIONAL RESULTS
# ============================================================

mlp_results = pd.DataFrame({

    "Model": [
        "Lightweight MLP"
    ],

    "Accuracy": [
        accuracy
    ],

    "Precision": [
        precision
    ],

    "Recall": [
        recall
    ],

    "F1_Score": [
        f1
    ],

    "Training_Time_Seconds": [
        training_time
    ],

    "Training_Time_Minutes": [
        training_time / 60
    ],

    "Inference_Time_Seconds": [
        inference_time
    ],

    "Latency_Per_Sample_ms": [
        latency_ms
    ],

    "Model_Size_MB": [
        model_size_mb
    ],

    "Parameters": [
        total_parameters
    ],

    "Training_Samples": [
        len(X_train)
    ],

    "Test_Samples": [
        len(X_test)
    ],

    "Features": [
        num_features
    ],

    "Classes": [
        num_classes
    ]

})

results_path = os.path.join(

    RESULTS_DIR,

    "mlp_model_results.csv"

)

mlp_results.to_csv(
    results_path,
    index=False
)

print("\nMLP computational results saved:")

print(results_path)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 10B COMPLETED")
print("=" * 70)

print("\nGenerated files:")

print("- lightweight_mlp_ids.keras")
print("- mlp_model_results.csv")
print("- mlp_classification_report.txt")
print("- mlp_confusion_matrix.csv")
print("- mlp_training_history.csv")

print("\nImportant:")
print("- MLP was trained only as a lightweight DL baseline.")
print("- Original four-classifier hybrid model was not modified.")
print("- Original test set remained untouched.")
print("- Computational cost and latency were measured.")

print("\nNext step:")
print("Compare ML hybrid vs lightweight MLP.")
print("=" * 70)
