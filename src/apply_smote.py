# ==========================================
# STEP 6B - CONTROLLED SMOTE
# CICIDS2017 Hybrid ML IDS
# ==========================================

import pandas as pd
import numpy as np
import joblib
import time
import gc

from pathlib import Path

from imblearn.over_sampling import SMOTE


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


X_TRAIN_FILE = (
    PROCESSED_DATA_DIR /
    "X_train_selected.csv"
)

Y_TRAIN_FILE = (
    PROCESSED_DATA_DIR /
    "y_train.csv"
)


X_SMOTE_FILE = (
    PROCESSED_DATA_DIR /
    "X_train_smote.csv"
)

Y_SMOTE_FILE = (
    PROCESSED_DATA_DIR /
    "y_train_smote.csv"
)

SMOTE_STRATEGY_FILE = (
    PROCESSED_DATA_DIR /
    "smote_strategy.pkl"
)


# ==========================================
# 2. SMOTE CONFIGURATION
# ==========================================

SMOTE_TARGET = 20_000

RANDOM_STATE = 42

# Default SMOTE uses 5 nearest neighbours.
# We keep this consistent with the standard
# SMOTE configuration unless it becomes
# impossible for a class.
K_NEIGHBORS = 5


# ==========================================
# 3. CHECK INPUT FILES
# ==========================================

if not X_TRAIN_FILE.exists():

    print("ERROR: X_train_selected.csv not found.")

    print("\nComplete Step 5 first.")

    exit()


if not Y_TRAIN_FILE.exists():

    print("ERROR: y_train.csv not found.")

    print("\nComplete Step 4 first.")

    exit()


# ==========================================
# 4. START
# ==========================================

print("=" * 70)
print("STEP 6B - CONTROLLED SMOTE")
print("=" * 70)

start_time = time.time()


# ==========================================
# 5. LOAD DATA
# ==========================================

print("\nLoading selected training features...")

X_train = pd.read_csv(
    X_TRAIN_FILE,
    low_memory=False
)

print(
    "X_train shape:",
    X_train.shape
)


print("\nLoading training labels...")

y_train = pd.read_csv(
    Y_TRAIN_FILE
)["Label"]

print(
    "y_train shape:",
    y_train.shape
)


# ==========================================
# 6. ORIGINAL CLASS DISTRIBUTION
# ==========================================

original_counts = (
    y_train
    .value_counts()
    .sort_index()
)

print("\n" + "=" * 70)
print("ORIGINAL TRAINING DISTRIBUTION")
print("=" * 70)

for class_id, count in original_counts.items():

    print(
        f"Class {class_id:2d} : {count:,}"
    )


# ==========================================
# 7. CREATE SAMPLING STRATEGY
# ==========================================

sampling_strategy = {}

for class_id, count in original_counts.items():

    # Do not oversample classes already
    # above the target.
    if count < SMOTE_TARGET:

        sampling_strategy[class_id] = SMOTE_TARGET


# ==========================================
# 8. DISPLAY STRATEGY
# ==========================================

print("\n" + "=" * 70)
print("SMOTE SAMPLING STRATEGY")
print("=" * 70)

print(
    f"\nTarget for minority classes:"
    f" {SMOTE_TARGET:,}"
)

for class_id, target in sampling_strategy.items():

    current = original_counts[class_id]

    print(
        f"Class {class_id:2d} | "
        f"{current:>10,} → {target:>10,}"
    )


# ==========================================
# 9. SAFETY CHECK
# ==========================================

minimum_class_count = min(
    original_counts[class_id]
    for class_id in sampling_strategy
)

print("\nSmallest class count:")
print(minimum_class_count)


if minimum_class_count < K_NEIGHBORS + 1:

    print(
        "\nWARNING:"
    )

    print(
        "At least one class has too few samples "
        "for the configured SMOTE k_neighbors."
    )

    print(
        "SMOTE will not be executed."
    )

    exit()


# ==========================================
# 10. ESTIMATE SIZE
# ==========================================

additional_samples = sum(
    sampling_strategy[class_id]
    -
    original_counts[class_id]

    for class_id in sampling_strategy
)

estimated_size = (
    len(y_train)
    +
    additional_samples
)


print("\n" + "=" * 70)
print("SIZE ESTIMATION")
print("=" * 70)

print(
    f"\nOriginal samples : "
    f"{len(y_train):,}"
)

print(
    f"New synthetic samples : "
    f"{additional_samples:,}"
)

print(
    f"Estimated final samples : "
    f"{estimated_size:,}"
)


# ==========================================
# 11. APPLY SMOTE
# ==========================================

print("\n" + "=" * 70)
print("APPLYING SMOTE")
print("=" * 70)

print("\nPlease wait...")

smote = SMOTE(
    sampling_strategy=sampling_strategy,
    random_state=RANDOM_STATE,
    k_neighbors=K_NEIGHBORS
)


X_train_smote, y_train_smote = (
    smote.fit_resample(
        X_train,
        y_train
    )
)


# ==========================================
# 12. CHECK RESULT
# ==========================================

print("\nSMOTE completed successfully.")

print("\nOriginal shape:")
print(
    X_train.shape,
    y_train.shape
)

print("\nAfter SMOTE:")
print(
    X_train_smote.shape,
    y_train_smote.shape
)


# ==========================================
# 13. NEW CLASS DISTRIBUTION
# ==========================================

new_counts = (
    pd.Series(y_train_smote)
    .value_counts()
    .sort_index()
)

print("\n" + "=" * 70)
print("POST-SMOTE CLASS DISTRIBUTION")
print("=" * 70)

for class_id, count in new_counts.items():

    print(
        f"Class {class_id:2d} : {count:,}"
    )


# ==========================================
# 14. SAVE SMOTE DATA
# ==========================================

print("\n" + "=" * 70)
print("SAVING SMOTE DATA")
print("=" * 70)

X_train_smote_df = pd.DataFrame(
    X_train_smote,
    columns=X_train.columns
)

y_train_smote_df = pd.DataFrame({
    "Label": y_train_smote
})


print("\nSaving:")
print(
    "-",
    X_SMOTE_FILE.name
)

print(
    "-",
    Y_SMOTE_FILE.name
)


X_train_smote_df.to_csv(
    X_SMOTE_FILE,
    index=False
)

y_train_smote_df.to_csv(
    Y_SMOTE_FILE,
    index=False
)


# ==========================================
# 15. SAVE SMOTE CONFIGURATION
# ==========================================

smote_configuration = {

    "method": "SMOTE",

    "target_count": SMOTE_TARGET,

    "random_state": RANDOM_STATE,

    "k_neighbors": K_NEIGHBORS,

    "sampling_strategy": sampling_strategy

}


joblib.dump(
    smote_configuration,
    SMOTE_STRATEGY_FILE
)


# ==========================================
# 16. CLEAN MEMORY
# ==========================================

del X_train
del y_train
del X_train_smote
del y_train_smote

gc.collect()


# ==========================================
# 17. FINAL
# ==========================================

elapsed_time = (
    time.time() - start_time
)


print("\n" + "=" * 70)
print("STEP 6B COMPLETED")
print("=" * 70)

print(
    f"\nProcessing time:"
    f" {elapsed_time / 60:.2f} minutes"
)

print(
    "\nSMOTE training data saved successfully."
)

print(
    "\nImportant:"
)

print(
    "SMOTE was applied ONLY to the training data."
)

print(
    "\nThe test set remains completely untouched."
)

print("=" * 70)
