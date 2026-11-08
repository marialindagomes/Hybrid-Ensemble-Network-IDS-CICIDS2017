# ==========================================
# STEP 8B
# FOUR-CLASSIFIER SOFT VOTING HYBRID MODEL
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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import time
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

HYBRID_MODEL_FILE = (
    MODELS_DIR /
    "hybrid_soft_voting_ids.pkl"
)

HYBRID_RESULTS_FILE = (
    RESULTS_DIR /
    "hybrid_model_results.csv"
)

HYBRID_TIMING_FILE = (
    RESULTS_DIR /
    "hybrid_model_timing.csv"
)

HYBRID_CM_FILE = (
    RESULTS_DIR /
    "hybrid_confusion_matrix.csv"
)


# ==========================================
# 4. START
# ==========================================

print("=" * 70)
print("STEP 8B - FOUR-CLASSIFIER SOFT VOTING HYBRID")
print("=" * 70)

start_total = time.perf_counter()


# ==========================================
# 5. CHECK INPUT FILES
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

print("\nClasses:")

for i, class_name in enumerate(class_names):

    print(
        f"{i:2d} -> {class_name}"
    )


# ==========================================
# 8. CONVERT LABELS
# ==========================================

y_train = y_train.astype(
    np.int32
)

y_test = y_test.astype(
    np.int32
)


# ==========================================
# 9. CREATE FOUR BASE CLASSIFIERS
# ==========================================

print("\n" + "=" * 70)
print("CREATING FOUR BASE CLASSIFIERS")
print("=" * 70)


rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)


lr = LogisticRegression(
    max_iter=3000,
    n_jobs=-1,
    random_state=42
)


dt = DecisionTreeClassifier(
    random_state=42
)


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


# ==========================================
# 10. CREATE SOFT VOTING HYBRID
# ==========================================

print("\nCreating Soft Voting classifier...")

hybrid_model = VotingClassifier(

    estimators=[
        ("rf", rf),
        ("lr", lr),
        ("dt", dt),
        ("xgb", xgb)
    ],

    voting="soft",

    n_jobs=-1
)


print("\nHybrid configuration:")

print(
    "RF  + LR  + DT  + XGBoost"
)

print(
    "Voting method: Soft Voting"
)


# ==========================================
# 11. TRAIN HYBRID
# ==========================================

print("\n" + "=" * 70)
print("TRAINING FOUR-CLASSIFIER HYBRID")
print("=" * 70)

print("\nTraining model...")

train_start = time.perf_counter()

hybrid_model.fit(
    X_train,
    y_train
)

train_end = time.perf_counter()

training_time = (
    train_end -
    train_start
)

print(
    f"\nTraining time:"
    f" {training_time:.4f} seconds"
)

print(
    f"Training time:"
    f" {training_time / 60:.2f} minutes"
)


# ==========================================
# 12. SAVE HYBRID MODEL
# ==========================================

print("\nSaving hybrid model...")

joblib.dump(
    hybrid_model,
    HYBRID_MODEL_FILE
)

print(
    "Model saved:"
)

print(
    HYBRID_MODEL_FILE
)


# ==========================================
# 13. MODEL SIZE
# ==========================================

model_size_mb = (
    HYBRID_MODEL_FILE.stat().st_size
    /
    (1024 ** 2)
)

print(
    f"\nSaved model size:"
    f" {model_size_mb:.2f} MB"
)


# ==========================================
# 14. PREDICTION
# ==========================================

print("\n" + "=" * 70)
print("HYBRID PREDICTION")
print("=" * 70)

print("\nGenerating predictions...")

prediction_start = time.perf_counter()

pred = hybrid_model.predict(
    X_test
)

prediction_end = time.perf_counter()

inference_time = (
    prediction_end -
    prediction_start
)


# ==========================================
# 15. LATENCY
# ==========================================

test_samples = len(
    X_test
)

latency_per_sample_ms = (
    inference_time /
    test_samples
) * 1000


# ==========================================
# 16. METRICS
# ==========================================

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


# ==========================================
# 17. DISPLAY RESULTS
# ==========================================

print("\n" + "=" * 70)
print("HYBRID MODEL RESULTS")
print("=" * 70)

print(
    f"\nAccuracy  : {accuracy:.6f}"
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
    f"\nInference time:"
    f" {inference_time:.4f} seconds"
)

print(
    f"Latency/sample:"
    f" {latency_per_sample_ms:.6f} ms"
)

print(
    f"Model size:"
    f" {model_size_mb:.2f} MB"
)


# ==========================================
# 18. CLASSIFICATION REPORT
# ==========================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        pred,
        target_names=class_names,
        zero_division=0
    )
)


# ==========================================
# 19. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    pred
)

cm_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
)

cm_df.to_csv(
    HYBRID_CM_FILE
)

print(
    "\nConfusion matrix saved:"
)

print(
    HYBRID_CM_FILE
)


# ==========================================
# 20. SAVE PERFORMANCE RESULTS
# ==========================================

results_df = pd.DataFrame([{

    "Model":
        "Four-Class Soft Voting Hybrid",

    "Accuracy":
        accuracy,

    "Precision":
        precision,

    "Recall":
        recall,

    "F1_Score":
        f1

}])


results_df.to_csv(
    HYBRID_RESULTS_FILE,
    index=False
)


# ==========================================
# 21. SAVE COMPUTATIONAL RESULTS
# ==========================================

timing_df = pd.DataFrame([{

    "Model":
        "Four-Class Soft Voting Hybrid",

    "Training_Time_Seconds":
        training_time,

    "Training_Time_Minutes":
        training_time / 60,

    "Inference_Time_Seconds":
        inference_time,

    "Latency_Per_Sample_ms":
        latency_per_sample_ms,

    "Model_Size_MB":
        model_size_mb,

    "Test_Samples":
        test_samples

}])


timing_df.to_csv(
    HYBRID_TIMING_FILE,
    index=False
)


# ==========================================
# 22. FINAL
# ==========================================

total_time = (
    time.perf_counter()
    -
    start_total
)


print("\n" + "=" * 70)
print("STEP 8B COMPLETED")
print("=" * 70)

print(
    f"\nTotal processing time:"
    f" {total_time / 60:.2f} minutes"
)

print("\nGenerated files:")

print(
    "- hybrid_soft_voting_ids.pkl"
)

print(
    "- hybrid_model_results.csv"
)

print(
    "- hybrid_model_timing.csv"
)

print(
    "- hybrid_confusion_matrix.csv"
)

print("\nNext step:")

print(
    "Ablation Study"
)

print(
    "Individual → 2-model → 3-model → "
    "4-model combinations"
)

print("=" * 70)
