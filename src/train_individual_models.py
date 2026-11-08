# ==========================================
# STEP 8A
# INDIVIDUAL CLASSIFIER TRAINING
# CICIDS2017 Hybrid ML IDS
# ==========================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import time
import json
import warnings

warnings.filterwarnings("ignore")


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DATA_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

MODELS_DIR = (
    PROJECT_ROOT / "models"
)

RESULTS_DIR = (
    PROJECT_ROOT / "results"
)

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================
# 2. INPUT FILES
# ==========================================

X_TRAIN_FILE = (
    PROCESSED_DATA_DIR /
    "X_train_scaled.csv"
)

X_TEST_FILE = (
    PROCESSED_DATA_DIR /
    "X_test_scaled.csv"
)

Y_TRAIN_FILE = (
    PROCESSED_DATA_DIR /
    "y_train_smote.csv"
)

Y_TEST_FILE = (
    PROCESSED_DATA_DIR /
    "y_test.csv"
)

LABEL_ENCODER_FILE = (
    PROCESSED_DATA_DIR /
    "label_encoder.pkl"
)


# ==========================================
# 3. OUTPUT FILES
# ==========================================

RESULTS_FILE = (
    RESULTS_DIR /
    "individual_model_results.csv"
)

TIMING_FILE = (
    RESULTS_DIR /
    "individual_model_timing.csv"
)


# ==========================================
# 4. START
# ==========================================

print("=" * 70)
print("STEP 8A - INDIVIDUAL CLASSIFIER TRAINING")
print("=" * 70)


start_total = time.perf_counter()


# ==========================================
# 5. CHECK FILES
# ==========================================

required_files = [
    X_TRAIN_FILE,
    X_TEST_FILE,
    Y_TRAIN_FILE,
    Y_TEST_FILE,
    LABEL_ENCODER_FILE
]

for file in required_files:

    if not file.exists():

        print("\nERROR: Required file not found:")
        print(file)

        print(
            "\nPlease complete previous steps first."
        )

        exit()


# ==========================================
# 6. LOAD DATA
# ==========================================

print("\nLoading scaled training data...")

X_train = pd.read_csv(
    X_TRAIN_FILE,
    low_memory=False
)

print(
    "X_train shape:",
    X_train.shape
)


print("\nLoading scaled test data...")

X_test = pd.read_csv(
    X_TEST_FILE,
    low_memory=False
)

print(
    "X_test shape:",
    X_test.shape
)


print("\nLoading training labels...")

y_train = pd.read_csv(
    Y_TRAIN_FILE
)["Label"]

print(
    "y_train shape:",
    y_train.shape
)


print("\nLoading test labels...")

y_test = pd.read_csv(
    Y_TEST_FILE
)["Label"]

print(
    "y_test shape:",
    y_test.shape
)


# ==========================================
# 7. LOAD LABEL ENCODER
# ==========================================

label_encoder = joblib.load(
    LABEL_ENCODER_FILE
)

class_names = label_encoder.classes_

print("\nNumber of classes:")
print(len(class_names))

print("\nClasses:")

for i, class_name in enumerate(class_names):

    print(
        f"{i:2d} -> {class_name}"
    )


# ==========================================
# 8. CONVERT LABELS
# ==========================================
#
# y_train and y_test were already encoded
# during Step 4.
#
# We keep them numeric for model training.
#
# ==========================================

y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)


# ==========================================
# 9. DEFINE FOUR MODELS
# ==========================================

print("\n" + "=" * 70)
print("CREATING INDIVIDUAL CLASSIFIERS")
print("=" * 70)


# ------------------------------------------
# Logistic Regression
# ------------------------------------------

lr = LogisticRegression(
    max_iter=3000,
    n_jobs=-1,
    random_state=42
)


# ------------------------------------------
# Decision Tree
# ------------------------------------------

dt = DecisionTreeClassifier(
    random_state=42
)


# ------------------------------------------
# Random Forest
# ------------------------------------------

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)


# ------------------------------------------
# XGBoost
# ------------------------------------------

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)


models = {

    "Logistic Regression": lr,

    "Decision Tree": dt,

    "Random Forest": rf,

    "XGBoost": xgb

}


# ==========================================
# 10. RESULT STORAGE
# ==========================================

results = []

timing_results = []


# ==========================================
# 11. TRAIN EACH MODEL
# ==========================================

for model_name, model in models.items():

    print("\n")
    print("=" * 70)
    print(f"TRAINING: {model_name}")
    print("=" * 70)

    # --------------------------------------
    # Training time
    # --------------------------------------

    print("\nTraining model...")

    train_start = time.perf_counter()

    model.fit(
        X_train,
        y_train
    )

    train_end = time.perf_counter()

    training_time = (
        train_end -
        train_start
    )

    print(
        f"Training time: "
        f"{training_time:.4f} seconds"
    )

    print(
        f"Training time: "
        f"{training_time / 60:.2f} minutes"
    )

    # --------------------------------------
    # Save model
    # --------------------------------------

    model_filename = (
        model_name
        .lower()
        .replace(" ", "_")
        + ".pkl"
    )

    model_path = (
        MODELS_DIR /
        model_filename
    )

    joblib.dump(
        model,
        model_path
    )

    print(
        "\nModel saved:"
    )

    print(
        model_path
    )

    # --------------------------------------
    # Prediction
    # --------------------------------------

    print(
        "\nGenerating predictions..."
    )

    prediction_start = time.perf_counter()

    pred = model.predict(
        X_test
    )

    prediction_end = time.perf_counter()

    inference_time = (
        prediction_end -
        prediction_start
    )

    # --------------------------------------
    # Per-sample latency
    # --------------------------------------

    number_of_test_samples = len(
        X_test
    )

    latency_per_sample_ms = (
        inference_time /
        number_of_test_samples
    ) * 1000

    # --------------------------------------
    # Metrics
    # --------------------------------------

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

    # --------------------------------------
    # Print results
    # --------------------------------------

    print("\n" + "-" * 70)
    print(f"{model_name} RESULTS")
    print("-" * 70)

    print(
        f"Accuracy  : {accuracy:.6f}"
    )

    print(
        f"Precision : {precision:.6f}"
    )

    print(
        f"Recall    : {recall:.6f}"
    )

    print(
        f"F1 Score  : {f1:.6f}"
    )

    print(
        f"Inference time : "
        f"{inference_time:.4f} seconds"
    )

    print(
        f"Latency/sample : "
        f"{latency_per_sample_ms:.6f} ms"
    )

    # --------------------------------------
    # Store metric results
    # --------------------------------------

    results.append({

        "Model": model_name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1_Score": f1

    })

    # --------------------------------------
    # Store timing results
    # --------------------------------------

    timing_results.append({

        "Model": model_name,

        "Training_Time_Seconds":
            training_time,

        "Training_Time_Minutes":
            training_time / 60,

        "Inference_Time_Seconds":
            inference_time,

        "Latency_Per_Sample_ms":
            latency_per_sample_ms,

        "Test_Samples":
            number_of_test_samples

    })

    # --------------------------------------
    # Classification report
    # --------------------------------------

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            pred,
            target_names=class_names,
            zero_division=0
        )
    )

    # --------------------------------------
    # Confusion matrix
    # --------------------------------------

    cm = confusion_matrix(
        y_test,
        pred
    )

    cm_filename = (
        model_name
        .lower()
        .replace(" ", "_")
        + "_confusion_matrix.csv"
    )

    cm_path = (
        RESULTS_DIR /
        cm_filename
    )

    cm_df = pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names
    )

    cm_df.to_csv(
        cm_path
    )

    print(
        "Confusion matrix saved:"
    )

    print(
        cm_path
    )


# ==========================================
# 12. SAVE OVERALL RESULTS
# ==========================================

results_df = pd.DataFrame(
    results
)

timing_df = pd.DataFrame(
    timing_results
)


results_df.to_csv(
    RESULTS_FILE,
    index=False
)


timing_df.to_csv(
    TIMING_FILE,
    index=False
)


# ==========================================
# 13. DISPLAY FINAL COMPARISON
# ==========================================

print("\n")
print("=" * 70)
print("INDIVIDUAL MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


print("\n")
print("=" * 70)
print("COMPUTATIONAL COST / LATENCY")
print("=" * 70)

print(
    timing_df.to_string(
        index=False
    )
)


# ==========================================
# 14. FINAL
# ==========================================

total_time = (
    time.perf_counter()
    -
    start_total
)


print("\n")
print("=" * 70)
print("STEP 8A COMPLETED")
print("=" * 70)

print(
    f"\nTotal processing time:"
    f" {total_time / 60:.2f} minutes"
)

print("\nModels saved in:")

print(
    MODELS_DIR
)

print("\nResults saved in:")

print(
    RESULTS_DIR
)

print("\nGenerated result files:")

print(
    "- individual_model_results.csv"
)

print(
    "- individual_model_timing.csv"
)

print(
    "\nNext step:"
)

print(
    "Construct the four-classifier Soft Voting Hybrid model."
)

print("=" * 70)
