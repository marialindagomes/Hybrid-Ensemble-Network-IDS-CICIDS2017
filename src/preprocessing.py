# ==========================================
# STEP 3 - DATA MERGING AND CLEANING
# CICIDS2017 Hybrid ML IDS
# ==========================================

import pandas as pd
import numpy as np
from pathlib import Path
import gc
import time


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Create processed folder if it does not exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. FIND ALL CSV FILES
# ==========================================

csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))


if len(csv_files) == 0:
    print("ERROR: No CSV files found in data/raw/")
    exit()


print("=" * 70)
print("CICIDS2017 DATA CLEANING")
print("=" * 70)

print("\nNumber of files:", len(csv_files))


# ==========================================
# 3. LOAD ALL CSV FILES
# ==========================================

dataframes = []

total_rows_before = 0

start_time = time.time()

print("\nLoading CSV files...")
print("-" * 70)


for i, file in enumerate(csv_files, start=1):

    print(f"\n[{i}/{len(csv_files)}] Loading:")
    print(file.name)

    df = pd.read_csv(
        file,
        low_memory=False
    )

    # Remove spaces around column names
    df.columns = df.columns.str.strip()

    print("Rows loaded:", f"{len(df):,}")
    print("Columns    :", len(df.columns))

    total_rows_before += len(df)

    dataframes.append(df)

    print("File loaded successfully.")


# ==========================================
# 4. MERGE ALL DATASETS
# ==========================================

print("\n" + "=" * 70)
print("MERGING ALL CSV FILES")
print("=" * 70)

df = pd.concat(
    dataframes,
    ignore_index=True
)

# Release individual DataFrames
del dataframes
gc.collect()


print("\nMerged dataset shape:")
print(df.shape)

print("\nTotal rows before cleaning:")
print(f"{len(df):,}")


# ==========================================
# 5. CHECK TARGET COLUMN
# ==========================================

target_col = "Label"

if target_col not in df.columns:

    print("\nERROR: Label column not found!")

    print("\nAvailable columns:")
    print(df.columns.tolist())

    exit()


print("\nTarget column:", target_col)


# ==========================================
# 6. REMOVE DUPLICATE ROWS
# ==========================================

print("\n" + "=" * 70)
print("STEP 1 - REMOVING DUPLICATE ROWS")
print("=" * 70)

rows_before_duplicates = len(df)

df.drop_duplicates(
    inplace=True
)

rows_after_duplicates = len(df)

duplicates_removed = (
    rows_before_duplicates -
    rows_after_duplicates
)


print("\nRows before duplicate removal:")
print(f"{rows_before_duplicates:,}")

print("\nDuplicate rows removed:")
print(f"{duplicates_removed:,}")

print("\nRows after duplicate removal:")
print(f"{rows_after_duplicates:,}")


# ==========================================
# 7. REPLACE INFINITY WITH NaN
# ==========================================

print("\n" + "=" * 70)
print("STEP 2 - HANDLING INFINITE VALUES")
print("=" * 70)

# Count infinity values before replacement
numeric_columns = df.select_dtypes(
    include=[np.number]
).columns

infinity_count = np.isinf(
    df[numeric_columns]
).sum().sum()

print("\nInfinite values found:")
print(f"{infinity_count:,}")


# Replace positive and negative infinity with NaN
df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

print("\nInfinite values replaced with NaN.")


# ==========================================
# 8. REMOVE MISSING VALUES
# ==========================================

print("\n" + "=" * 70)
print("STEP 3 - REMOVING MISSING VALUES")
print("=" * 70)

rows_before_nan = len(df)

missing_rows = df.isna().any(axis=1).sum()

print("\nRows containing missing values:")
print(f"{missing_rows:,}")


# Remove rows containing NaN
df.dropna(
    inplace=True
)

rows_after_nan = len(df)

nan_rows_removed = (
    rows_before_nan -
    rows_after_nan
)


print("\nRows removed because of missing values:")
print(f"{nan_rows_removed:,}")

print("\nRows after missing-value removal:")
print(f"{rows_after_nan:,}")


# ==========================================
# 9. CLEAN LABEL VALUES
# ==========================================

print("\n" + "=" * 70)
print("CHECKING LABEL VALUES")
print("=" * 70)

# Remove accidental spaces around labels
df[target_col] = (
    df[target_col]
    .astype(str)
    .str.strip()
)

print("\nUnique classes after cleaning:")

class_counts = df[target_col].value_counts()

for label, count in class_counts.items():

    print(
        f"{label:<35} {count:,}"
    )


# ==========================================
# 10. FINAL DATASET INFORMATION
# ==========================================

print("\n" + "=" * 70)
print("FINAL CLEANED DATASET")
print("=" * 70)

print("\nFinal shape:")
print(df.shape)

print("\nFinal number of rows:")
print(f"{len(df):,}")

print("\nFinal number of columns:")
print(len(df.columns))

print("\nRows removed during cleaning:")
print(
    f"{total_rows_before - len(df):,}"
)


# ==========================================
# 11. SAVE CLEANED DATASET
# ==========================================

output_file = (
    PROCESSED_DATA_DIR /
    "cicids2017_cleaned.csv"
)

print("\n" + "=" * 70)
print("SAVING CLEANED DATASET")
print("=" * 70)

print("\nOutput file:")
print(output_file)

df.to_csv(
    output_file,
    index=False
)

print("\nCleaned dataset saved successfully!")


# ==========================================
# 12. MEMORY CLEANUP
# ==========================================

del df
gc.collect()


# ==========================================
# 13. FINAL MESSAGE
# ==========================================

elapsed_time = time.time() - start_time

print("\n" + "=" * 70)
print("STEP 3 COMPLETED")
print("=" * 70)

print(
    f"\nProcessing time: {elapsed_time / 60:.2f} minutes"
)

print("\nNext step:")
print("Feature/target separation → Train/Validation/Test split")
print("=" * 70)
