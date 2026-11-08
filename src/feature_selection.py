# ==========================================
# STEP 5 - FEATURE SELECTION
# SelectKBest + ANOVA F-test
# CICIDS2017 Hybrid ML IDS
# ==========================================

import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.feature_selection import SelectKBest, f_classif


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


# ==========================================
# 2. INPUT FILES
# ==========================================

X_TRAIN_FILE = PROCESSED_DATA_DIR / "X_train.csv"
X_TEST_FILE = PROCESSED_DATA_DIR / "X_test.csv"

Y_TRAIN_FILE = PROCESSED_DATA_DIR / "y_train.csv"


# ==========================================
# 3. OUTPUT FILES
# ==========================================

X_TRAIN_SELECTED_FILE = (
    PROCESSED_DATA_DIR /
    "X_train_selected.csv"
)

X_TEST_SELECTED_FILE = (
    PROCESSED_DATA_DIR /
    "X_test_selected.csv"
)

SELECTOR_FILE = (
    PROCESSED_DATA_DIR /
    "selectkbest_selector.pkl"
)

SELECTED_FEATURES_FILE = (
    PROCESSED_DATA_DIR /
    "selected_features.csv"
)


# ==========================================
# 4. CHECK INPUT FILES
# ==========================================

required_files = [
    X_TRAIN_FILE,
    X_TEST_FILE,
    Y_TRAIN_FILE
]

for file in required_files:

    if not file.exists():

        print("ERROR: Required file not found:")
        print(file)

        print("\nPlease complete Step 4 first.")

        exit()


# ==========================================
# 5. LOAD TRAINING DATA
# ==========================================

print("=" * 70)
print("STEP 5 - FEATURE SELECTION")
print("=" * 70)

print("\nLoading training features...")

X_train = pd.read_csv(
    X_TRAIN_FILE,
    low_memory=False
)

print("X_train shape:")
print(X_train.shape)


print("\nLoading test features...")

X_test = pd.read_csv(
    X_TEST_FILE,
    low_memory=False
)

print("X_test shape:")
print(X_test.shape)


print("\nLoading training labels...")

y_train = pd.read_csv(
    Y_TRAIN_FILE
)["Label"]

print("y_train shape:")
print(y_train.shape)


# ==========================================
# 6. CHECK FEATURE COUNT
# ==========================================

original_feature_count = X_train.shape[1]

print("\nOriginal number of features:")
print(original_feature_count)


# ==========================================
# 7. SELECT TOP 30 FEATURES
# ==========================================

K = 30

print("\n" + "=" * 70)
print("SELECTING TOP 30 FEATURES")
print("=" * 70)

print("\nMethod:")
print("SelectKBest")

print("\nScoring function:")
print("ANOVA F-test (f_classif)")

print("\nNumber of selected features:")
print(K)


selector = SelectKBest(
    score_func=f_classif,
    k=K
)


# IMPORTANT:
# Fit selector ONLY on training data

print("\nFitting SelectKBest on training data...")

X_train_selected = selector.fit_transform(
    X_train,
    y_train
)


# Apply the already-fitted selector to test data

print("Transforming test data...")

X_test_selected = selector.transform(
    X_test
)


# ==========================================
# 8. GET SELECTED FEATURE NAMES
# ==========================================

selected_mask = selector.get_support()

selected_features = X_train.columns[
    selected_mask
]


# ==========================================
# 9. DISPLAY SELECTED FEATURES
# ==========================================

print("\n" + "=" * 70)
print("SELECTED FEATURES")
print("=" * 70)

for i, feature in enumerate(
    selected_features,
    start=1
):

    print(
        f"{i:2d}. {feature}"
    )


# ==========================================
# 10. DISPLAY F-SCORES
# ==========================================

print("\n" + "=" * 70)
print("FEATURE SCORES")
print("=" * 70)

feature_scores = selector.scores_

feature_score_table = pd.DataFrame({

    "Feature":
        X_train.columns,

    "F_Score":
        feature_scores,

    "Selected":
        selected_mask

})


feature_score_table = (
    feature_score_table
    .sort_values(
        by="F_Score",
        ascending=False
    )
)


print(
    feature_score_table.head(30).to_string(
        index=False
    )
)


# ==========================================
# 11. CHECK SHAPES
# ==========================================

print("\n" + "=" * 70)
print("FEATURE SELECTION RESULTS")
print("=" * 70)

print("\nBefore feature selection:")

print(
    "X_train:",
    X_train.shape
)

print(
    "X_test :",
    X_test.shape
)


print("\nAfter feature selection:")

print(
    "X_train_selected:",
    X_train_selected.shape
)

print(
    "X_test_selected :",
    X_test_selected.shape
)


# ==========================================
# 12. CONVERT TO DATAFRAMES
# ==========================================

X_train_selected_df = pd.DataFrame(
    X_train_selected,
    columns=selected_features
)

X_test_selected_df = pd.DataFrame(
    X_test_selected,
    columns=selected_features
)


# ==========================================
# 13. SAVE SELECTED DATA
# ==========================================

print("\n" + "=" * 70)
print("SAVING FEATURE-SELECTION RESULTS")
print("=" * 70)


X_train_selected_df.to_csv(
    X_TRAIN_SELECTED_FILE,
    index=False
)


X_test_selected_df.to_csv(
    X_TEST_SELECTED_FILE,
    index=False
)


# ==========================================
# 14. SAVE SELECTOR
# ==========================================

joblib.dump(
    selector,
    SELECTOR_FILE
)


# ==========================================
# 15. SAVE FEATURE INFORMATION
# ==========================================

feature_score_table.to_csv(
    SELECTED_FEATURES_FILE,
    index=False
)


# ==========================================
# 16. FINAL MESSAGE
# ==========================================

print("\nFiles saved:")

print(
    " -",
    X_TRAIN_SELECTED_FILE.name
)

print(
    " -",
    X_TEST_SELECTED_FILE.name
)

print(
    " -",
    SELECTOR_FILE.name
)

print(
    " -",
    SELECTED_FEATURES_FILE.name
)


print("\n" + "=" * 70)
print("STEP 5 COMPLETED")
print("=" * 70)

print(
    "\n78 original features → 30 selected features"
)

print(
    "\nImportant:"
)

print(
    "SelectKBest was fitted ONLY on training data."
)

print(
    "\nNext step:"
)

print(
    "SMOTE on training data ONLY."
)

print("=" * 70)
