# ==========================================
# STEP 6A - SMOTE STRATEGY ANALYSIS
# CICIDS2017 Hybrid ML IDS
# ==========================================

import pandas as pd
from pathlib import Path


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


# ==========================================
# 2. LOAD TRAINING LABELS
# ==========================================

print("=" * 70)
print("STEP 6A - SMOTE STRATEGY ANALYSIS")
print("=" * 70)

print("\nLoading training labels...")

y_train = pd.read_csv(
    Y_TRAIN_FILE
)["Label"]


# ==========================================
# 3. CLASS DISTRIBUTION
# ==========================================

class_counts = (
    y_train
    .value_counts()
    .sort_index()
)

print("\n" + "=" * 70)
print("CURRENT TRAINING CLASS DISTRIBUTION")
print("=" * 70)

for class_id, count in class_counts.items():

    print(
        f"Class {class_id:2d}"
        f" : {count:,}"
    )


# ==========================================
# 4. DETERMINE MAJORITY CLASS
# ==========================================

majority_count = class_counts.max()

majority_class = class_counts.idxmax()

print("\n" + "=" * 70)
print("MAJORITY CLASS")
print("=" * 70)

print(
    f"\nMajority class : {majority_class}"
)

print(
    f"Majority count : {majority_count:,}"
)


# ==========================================
# 5. CONTROLLED SMOTE TARGET
# ==========================================
#
# We DO NOT oversample every class to the
# full majority-class size.
#
# Instead, minority classes are increased
# to a controlled target.
#
# This prevents an extremely large synthetic
# dataset while still addressing imbalance.
#
# ==========================================

SMOTE_TARGET = 100_000

print("\n" + "=" * 70)
print("CONTROLLED SMOTE TARGET")
print("=" * 70)

print(
    f"\nTarget count for minority classes:"
    f" {SMOTE_TARGET:,}"
)


# ==========================================
# 6. BUILD SAMPLING STRATEGY
# ==========================================

sampling_strategy = {}

print("\n" + "=" * 70)
print("PROPOSED SMOTE SAMPLING STRATEGY")
print("=" * 70)

for class_id, count in class_counts.items():

    # Majority class stays unchanged
    if class_id == majority_class:

        continue

    # Only oversample classes below target
    if count < SMOTE_TARGET:

        sampling_strategy[class_id] = SMOTE_TARGET


# ==========================================
# 7. DISPLAY STRATEGY
# ==========================================

print("\nClasses that will be oversampled:")

for class_id, target_count in sampling_strategy.items():

    current_count = class_counts[class_id]

    synthetic_needed = (
        target_count -
        current_count
    )

    print(
        f"Class {class_id:2d} | "
        f"Current: {current_count:>10,} | "
        f"Target: {target_count:>10,} | "
        f"New synthetic samples: {synthetic_needed:>10,}"
    )


# ==========================================
# 8. ESTIMATE FINAL TRAINING SIZE
# ==========================================

original_total = class_counts.sum()

additional_samples = sum(
    sampling_strategy[class_id] -
    class_counts[class_id]
    for class_id in sampling_strategy
)

estimated_total = (
    original_total +
    additional_samples
)


print("\n" + "=" * 70)
print("SIZE ESTIMATION")
print("=" * 70)

print(
    f"\nOriginal training samples : "
    f"{original_total:,}"
)

print(
    f"Additional synthetic samples : "
    f"{additional_samples:,}"
)

print(
    f"Estimated SMOTE training size : "
    f"{estimated_total:,}"
)


# ==========================================
# 9. FINAL MESSAGE
# ==========================================

print("\n" + "=" * 70)
print("STEP 6A COMPLETED")
print("=" * 70)

print(
    "\nNo SMOTE samples were generated."
)

print(
    "This step only analyzed the proposed strategy."
)

print(
    "\nNext step:"
)

print(
    "Apply controlled SMOTE to X_train_selected "
    "and y_train."
)

print("=" * 70)
