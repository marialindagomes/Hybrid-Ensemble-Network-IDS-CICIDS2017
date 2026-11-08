# ==========================================
# STEP 7 - FEATURE SCALING
# StandardScaler
# CICIDS2017 Hybrid ML IDS
# ==========================================

import pandas as pd
import numpy as np
import joblib
import time

from pathlib import Path

from sklearn.preprocessing import StandardScaler


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DATA_DIR = (
    PROJECT_ROOT / "data" / "processed"
)


# ==========================================
# 2. INPUT FILES
# ==========================================

X_TRAIN_SMOTE_FILE = (
    PROCESSED_DATA_DIR /
    "X_train_smote.csv"
)

X_TEST_SELECTED_FILE = (
    PROCESSED_DATA_DIR /
    "X_test_selected.csv"
)


# ==========================================
# 3. OUTPUT FILES
# ==========================================

X_TRAIN_SCALED_FILE = (
    PROCESSED_DATA_DIR /
    "X_train_scaled.csv"
)

X_TEST_SCALED_FILE = (
    PROCESSED_DATA_DIR /
    "X_test_scaled.csv"
)

SCALER_FILE = (
    PROCESSED_DATA_DIR /
    "standard_scaler.pkl"
)


# ==========================================
# 4. CHECK INPUT FILES
# ==========================================

required_files = [
    X_TRAIN_SMOTE_FILE,
    X_TEST_SELECTED_FILE
]

for file in required_files:

    if not file.exists():

        print(
            "\nERROR: Required file not found:"
        )

        print(file)

        print(
            "\nPlease complete the previous "
            "steps first."
        )

        exit()


# ==========================================
# 5. START
# ==========================================

print("=" * 70)
print("STEP 7 - FEATURE SCALING")
print("=" * 70)

start_time = time.time()


# ==========================================
# 6. LOAD TRAINING DATA
# ==========================================

print(
    "\nLoading SMOTE training features..."
)

X_train = pd.read_csv(
    X_TRAIN_SMOTE_FILE,
    low_memory=False
)

print(
    "X_train_smote shape:"
)

print(
    X_train.shape
)


# ==========================================
# 7. LOAD TEST DATA
# ==========================================

print(
    "\nLoading selected test features..."
)

X_test = pd.read_csv(
    X_TEST_SELECTED_FILE,
    low_memory=False
)

print(
    "X_test_selected shape:"
)

print(
    X_test.shape
)


# ==========================================
# 8. CHECK FEATURE CONSISTENCY
# ==========================================

print(
    "\nChecking feature consistency..."
)

if list(X_train.columns) != list(X_test.columns):

    print(
        "\nERROR:"
    )

    print(
        "Training and test feature columns "
        "do not match."
    )

    exit()


print(
    "Feature columns match successfully."
)


# ==========================================
# 9. CREATE STANDARD SCALER
# ==========================================

print(
    "\n" + "=" * 70
)

print(
    "CREATING STANDARD SCALER"
)

print(
    "=" * 70
)


scaler = StandardScaler()


# ==========================================
# 10. FIT ON TRAINING DATA ONLY
# ==========================================

print(
    "\nFitting StandardScaler "
    "on SMOTE training data..."
)

X_train_scaled = scaler.fit_transform(
    X_train
)


# ==========================================
# 11. TRANSFORM TEST DATA
# ==========================================

print(
    "\nTransforming test data..."
)

X_test_scaled = scaler.transform(
    X_test
)


# ==========================================
# 12. CHECK RESULTS
# ==========================================

print(
    "\n" + "=" * 70
)

print(
    "SCALING RESULTS"
)

print(
    "=" * 70
)

print(
    "\nTraining data:"
)

print(
    "Before scaling:",
    X_train.shape
)

print(
    "After scaling :",
    X_train_scaled.shape
)


print(
    "\nTest data:"
)

print(
    "Before scaling:",
    X_test.shape
)

print(
    "After scaling :",
    X_test_scaled.shape
)


# ==========================================
# 13. BASIC STATISTICAL CHECK
# ==========================================

print(
    "\n" + "=" * 70
)

print(
    "SCALING VALIDATION"
)

print(
    "=" * 70
)


train_mean = np.mean(
    X_train_scaled,
    axis=0
)

train_std = np.std(
    X_train_scaled,
    axis=0
)


print(
    "\nMean of scaled training features:"
)

print(
    np.round(
        train_mean,
        4
    )
)


print(
    "\nStandard deviation of scaled "
    "training features:"
)

print(
    np.round(
        train_std,
        4
    )
)


# ==========================================
# 14. SAVE SCALED DATA
# ==========================================

print(
    "\n" + "=" * 70
)

print(
    "SAVING SCALED DATA"
)

print(
    "=" * 70
)


X_train_scaled_df = pd.DataFrame(
    X_train_scaled,
    columns=X_train.columns
)


X_test_scaled_df = pd.DataFrame(
    X_test_scaled,
    columns=X_test.columns
)


print(
    "\nSaving:"
)

print(
    "-",
    X_TRAIN_SCALED_FILE.name
)

print(
    "-",
    X_TEST_SCALED_FILE.name
)

print(
    "-",
    SCALER_FILE.name
)


X_train_scaled_df.to_csv(
    X_TRAIN_SCALED_FILE,
    index=False
)


X_test_scaled_df.to_csv(
    X_TEST_SCALED_FILE,
    index=False
)


# ==========================================
# 15. SAVE SCALER
# ==========================================

joblib.dump(
    scaler,
    SCALER_FILE
)


# ==========================================
# 16. FINAL
# ==========================================

elapsed_time = (
    time.time() - start_time
)


print(
    "\n" + "=" * 70
)

print(
    "STEP 7 COMPLETED"
)

print(
    "=" * 70
)

print(
    f"\nProcessing time:"
    f" {elapsed_time / 60:.2f} minutes"
)

print(
    "\nStandardScaler was fitted ONLY "
    "on training data."
)

print(
    "Test data was transformed using "
    "the fitted training scaler."
)

print(
    "\nFiles saved successfully."
)

print(
    "\nNext step:"
)

print(
    "Train the four individual classifiers:"
)

print(
    "1. Logistic Regression"
)

print(
    "2. Decision Tree"
)

print(
    "3. Random Forest"
)

print(
    "4. XGBoost"
)

print("=" * 70)