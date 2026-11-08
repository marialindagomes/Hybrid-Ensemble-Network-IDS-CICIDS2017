# ============================================================
# STEP 10A - PREPARE LIGHTWEIGHT DEEP LEARNING DATA
# CICIDS2017 Hybrid ML IDS
# ============================================================

import os
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_FEATURES = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "X_train_scaled.csv"
)

TRAIN_LABELS = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "y_train_smote.csv"
)

OUTPUT_FEATURES = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "X_dl_train.csv"
)

OUTPUT_LABELS = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "y_dl_train.csv"
)

# Number of samples for lightweight DL experiment
DL_SAMPLE_SIZE = 500_000

RANDOM_STATE = 42


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STEP 10A - PREPARE LIGHTWEIGHT DEEP-LEARNING DATA")
print("=" * 70)

start_time = time.time()


# ============================================================
# CHECK FILES
# ============================================================

print("\nChecking required files...")

if not os.path.exists(TRAIN_FEATURES):
    raise FileNotFoundError(
        f"Training features not found:\n{TRAIN_FEATURES}"
    )

if not os.path.exists(TRAIN_LABELS):
    raise FileNotFoundError(
        f"Training labels not found:\n{TRAIN_LABELS}"
    )

print("Required files found successfully.")


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading scaled training features...")

X_train = pd.read_csv(TRAIN_FEATURES)

print("X_train shape:", X_train.shape)

print("\nLoading SMOTE training labels...")

y_train = pd.read_csv(TRAIN_LABELS).squeeze()

print("y_train shape:", y_train.shape)


# ============================================================
# VALIDATION
# ============================================================

if len(X_train) != len(y_train):
    raise ValueError(
        "Feature and label sample counts do not match."
    )

print("\nFeature-label consistency check passed.")


# ============================================================
# CURRENT DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("CURRENT TRAINING DISTRIBUTION")
print("=" * 70)

distribution = y_train.value_counts().sort_index()

for class_id, count in distribution.items():
    print(f"Class {class_id:2d} : {count:,}")


# ============================================================
# CREATE STRATIFIED SUBSET
# ============================================================

print("\n" + "=" * 70)
print("CREATING LIGHTWEIGHT DL SUBSET")
print("=" * 70)

print(f"\nRequested samples: {DL_SAMPLE_SIZE:,}")

if DL_SAMPLE_SIZE >= len(X_train):

    print("Requested size is greater than or equal to")
    print("the available training data.")
    print("Using the complete training dataset.")

    X_dl = X_train.copy()
    y_dl = y_train.copy()

else:

    print("\nCreating stratified subset...")
    print("Random state:", RANDOM_STATE)

    # train_test_split is used only to obtain a
    # stratified subset of the already prepared
    # training data.

    X_dl, _, y_dl, _ = train_test_split(
        X_train,
        y_train,
        train_size=DL_SAMPLE_SIZE,
        stratify=y_train,
        random_state=RANDOM_STATE
    )


# ============================================================
# RESET INDEX
# ============================================================

X_dl = X_dl.reset_index(drop=True)
y_dl = y_dl.reset_index(drop=True)


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n" + "=" * 70)
print("DL TRAINING DATA")
print("=" * 70)

print("\nX_dl shape:")
print(X_dl.shape)

print("\ny_dl shape:")
print(y_dl.shape)

print(f"\nTotal DL training samples: {len(X_dl):,}")
print(f"Number of features: {X_dl.shape[1]}")


# ============================================================
# CHECK CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("DL SUBSET CLASS DISTRIBUTION")
print("=" * 70)

dl_distribution = y_dl.value_counts().sort_index()

for class_id in distribution.index:

    count = dl_distribution.get(class_id, 0)

    print(
        f"Class {class_id:2d} : {count:,}"
    )


# ============================================================
# SAVE DATA
# ============================================================

print("\n" + "=" * 70)
print("SAVING DL TRAINING DATA")
print("=" * 70)

X_dl.to_csv(
    OUTPUT_FEATURES,
    index=False
)

y_dl.to_csv(
    OUTPUT_LABELS,
    index=False,
    header=True
)

print("\nSaved:")
print("-", OUTPUT_FEATURES)
print("-", OUTPUT_LABELS)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

saved_X = pd.read_csv(OUTPUT_FEATURES)
saved_y = pd.read_csv(OUTPUT_LABELS).squeeze()

print("\nSaved feature shape:", saved_X.shape)
print("Saved label shape  :", saved_y.shape)

if len(saved_X) == len(saved_y):
    print("\nFeature-label sample count: MATCH")
else:
    print("\nERROR: Feature-label sample count mismatch!")


# ============================================================
# COMPLETION
# ============================================================

elapsed = time.time() - start_time

print("\n" + "=" * 70)
print("STEP 10A COMPLETED")
print("=" * 70)

print(f"\nProcessing time: {elapsed:.2f} seconds")

print("\nGenerated files:")
print("- X_dl_train.csv")
print("- y_dl_train.csv")

print("\nImportant:")
print("This is a lightweight DL baseline dataset.")
print("The original ML training data was not modified.")
print("The original test set remains completely untouched.")

print("\nNext step:")
print("Train the lightweight MLP/DNN baseline.")
print("=" * 70)

