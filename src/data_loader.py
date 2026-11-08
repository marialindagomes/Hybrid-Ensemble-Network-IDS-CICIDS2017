# ==========================================
# STEP 2 - CICIDS2017 DATASET INSPECTION
# Hybrid ML IDS Research
# ==========================================

import pandas as pd
from pathlib import Path


# ==========================================
# 1. DEFINE RAW DATA FOLDER
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


# ==========================================
# 2. FIND ALL CSV FILES
# ==========================================

csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))


print("=" * 70)
print("CICIDS2017 DATASET INSPECTION")
print("=" * 70)

print("\nRaw data folder:")
print(RAW_DATA_DIR)

print("\nNumber of CSV files found:", len(csv_files))


if len(csv_files) == 0:
    print("\nERROR: No CSV files found!")
    print("Please check the data/raw folder.")
    exit()


# ==========================================
# 3. DISPLAY FILE NAMES
# ==========================================

print("\nCSV Files:")
print("-" * 70)

for i, file in enumerate(csv_files, start=1):
    print(f"{i}. {file.name}")


# ==========================================
# 4. INSPECT EACH CSV FILE
# ==========================================

all_labels = set()

print("\n" + "=" * 70)
print("FILE-BY-FILE INFORMATION")
print("=" * 70)


for file in csv_files:

    print("\n" + "-" * 70)
    print("File:", file.name)
    print("-" * 70)

    try:

        # Load one CSV at a time
        df = pd.read_csv(
            file,
            low_memory=False
        )

        # Remove accidental spaces from column names
        df.columns = df.columns.str.strip()

        print("Rows       :", df.shape[0])
        print("Columns    :", df.shape[1])

        # Check Label column
        if "Label" in df.columns:

            print("Label      : Found")

            labels = df["Label"].astype(str).str.strip()

            # Store labels
            all_labels.update(labels.unique())

            print("\nClasses in this file:")

            for label, count in labels.value_counts().items():
                print(f"  {label:<30} {count:,}")

        else:

            print("Label      : NOT FOUND")

            print("\nAvailable columns:")
            print(df.columns.tolist())

    except Exception as e:

        print("ERROR while reading file:")
        print(e)


# ==========================================
# 5. DISPLAY ALL UNIQUE CLASSES
# ==========================================

print("\n" + "=" * 70)
print("ALL UNIQUE ATTACK CLASSES")
print("=" * 70)

all_labels = sorted(all_labels)

for i, label in enumerate(all_labels, start=1):
    print(f"{i}. {label}")


# ==========================================
# 6. FINAL SUMMARY
# ==========================================

print("\n" + "=" * 70)
print("DATASET INSPECTION SUMMARY")
print("=" * 70)

print("CSV files found :", len(csv_files))
print("Unique classes  :", len(all_labels))

print("\nDataset inspection completed successfully.")
print("=" * 70)
