# ==========================================
# STEP 4 - TRAIN / TEST SPLIT
# CICIDS2017 Hybrid ML IDS
# ==========================================

import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


INPUT_FILE = (
    PROCESSED_DATA_DIR /
    "cicids2017_cleaned.csv"
)


# ==========================================
# 2. OUTPUT FILES
# ==========================================

X_TRAIN_FILE = PROCESSED_DATA_DIR / "X_train.csv"
X_TEST_FILE = PROCESSED_DATA_DIR / "X_test.csv"

Y_TRAIN_FILE = PROCESSED_DATA_DIR / "y_train.csv"
Y_TEST_FILE = PROCESSED_DATA_DIR / "y_test.csv"

LABEL_ENCODER_FILE = (
    PROCESSED_DATA_DIR /
    "label_encoder.pkl"
)


# ==========================================
# 3. CHECK INPUT FILE
# ==========================================

if not INPUT_FILE.exists():

    print("ERROR: Cleaned dataset not found!")

    print("\nExpected file:")
    print(INPUT_FILE)

    print("\nRun Step 3 first.")

    exit()


# ==========================================
# 4. LOAD CLEANED DATASET
# ==========================================

print("=" * 70)
print("STEP 4 - TRAIN / TEST SPLIT")
print("=" * 70)

print("\nLoading cleaned dataset...")

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print("\nDataset shape:")
print(df.shape)


# ==========================================
# 5. TARGET COLUMN
# ==========================================

target_col = "Label"

if target_col not in df.columns:

    print("\nERROR: Label column not found!")

    exit()


# ==========================================
# 6. SEPARATE FEATURES AND TARGET
# ==========================================

print("\n" + "=" * 70)
print("SEPARATING FEATURES AND TARGET")
print("=" * 70)

X = df.drop(
    columns=[target_col]
)

y = df[target_col].astype(str).str.strip()


print("\nFeature shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)


# ==========================================
# 7. CONVERT FEATURES TO NUMERIC
# ==========================================

print("\n" + "=" * 70)
print("CONVERTING FEATURES TO NUMERIC")
print("=" * 70)

X = X.apply(
    pd.to_numeric,
    errors="coerce"
)

# Any unexpected conversion failures
# are filled with 0.
X.fillna(
    0,
    inplace=True
)

print("\nAll feature columns converted to numeric.")


# ==========================================
# 8. LABEL ENCODING
# ==========================================

print("\n" + "=" * 70)
print("LABEL ENCODING")
print("=" * 70)

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)


print("\nClasses:")

for index, label in enumerate(
    label_encoder.classes_
):

    print(
        f"{index:2d} -> {label}"
    )


print(
    "\nTotal classes:",
    len(label_encoder.classes_)
)


# ==========================================
# 9. TRAIN / TEST SPLIT
# ==========================================

print("\n" + "=" * 70)
print("CREATING TRAIN / TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y_encoded,

    test_size=0.20,

    random_state=42,

    stratify=y_encoded
)


# ==========================================
# 10. DISPLAY SPLIT INFORMATION
# ==========================================

print("\nTraining set:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTest set:")
print("X_test :", X_test.shape)
print("y_test :", y_test.shape)


# ==========================================
# 11. CHECK CLASS DISTRIBUTION
# ==========================================

print("\n" + "=" * 70)
print("TRAINING CLASS DISTRIBUTION")
print("=" * 70)

train_counts = pd.Series(
    y_train
).value_counts().sort_index()

for class_id, count in train_counts.items():

    label = label_encoder.inverse_transform(
        [class_id]
    )[0]

    print(
        f"{class_id:2d} | "
        f"{label:<35} | "
        f"{count:,}"
    )


print("\n" + "=" * 70)
print("TEST CLASS DISTRIBUTION")
print("=" * 70)

test_counts = pd.Series(
    y_test
).value_counts().sort_index()

for class_id, count in test_counts.items():

    label = label_encoder.inverse_transform(
        [class_id]
    )[0]

    print(
        f"{class_id:2d} | "
        f"{label:<35} | "
        f"{count:,}"
    )


# ==========================================
# 12. SAVE TRAIN / TEST DATA
# ==========================================

print("\n" + "=" * 70)
print("SAVING TRAIN / TEST DATA")
print("=" * 70)

X_train.to_csv(
    X_TRAIN_FILE,
    index=False
)

X_test.to_csv(
    X_TEST_FILE,
    index=False
)

pd.DataFrame({
    "Label": y_train
}).to_csv(
    Y_TRAIN_FILE,
    index=False
)

pd.DataFrame({
    "Label": y_test
}).to_csv(
    Y_TEST_FILE,
    index=False
)


# ==========================================
# 13. SAVE LABEL ENCODER
# ==========================================

joblib.dump(
    label_encoder,
    LABEL_ENCODER_FILE
)


# ==========================================
# 14. FINAL SUMMARY
# ==========================================

print("\nFiles saved:")

print(
    " -",
    X_TRAIN_FILE.name
)

print(
    " -",
    X_TEST_FILE.name
)

print(
    " -",
    Y_TRAIN_FILE.name
)

print(
    " -",
    Y_TEST_FILE.name
)

print(
    " -",
    LABEL_ENCODER_FILE.name
)


print("\n" + "=" * 70)
print("STEP 4 COMPLETED")
print("=" * 70)

print("\nThe test set has been kept completely separate.")

print(
    "\nNext step:"
    "\nFeature Selection (SelectKBest, top 30)"
    "\nusing TRAINING DATA ONLY."
)

print("=" * 70)
