# ============================================================
# STEP 11A - ML HYBRID VS LIGHTWEIGHT MLP COMPARISON
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)

# ============================================================
# FILE PATHS
# ============================================================

individual_results_file = os.path.join(
    RESULTS_DIR,
    "individual_model_results.csv"
)

individual_timing_file = os.path.join(
    RESULTS_DIR,
    "individual_model_timing.csv"
)

hybrid_results_file = os.path.join(
    RESULTS_DIR,
    "hybrid_model_results.csv"
)

hybrid_timing_file = os.path.join(
    RESULTS_DIR,
    "hybrid_model_timing.csv"
)

mlp_results_file = os.path.join(
    RESULTS_DIR,
    "mlp_model_results.csv"
)

# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STEP 11A - ML HYBRID VS LIGHTWEIGHT MLP COMPARISON")
print("=" * 70)

# ============================================================
# CHECK FILES
# ============================================================

print("\nChecking required files...")

required_files = [
    individual_results_file,
    individual_timing_file,
    hybrid_results_file,
    hybrid_timing_file,
    mlp_results_file
]

for file in required_files:

    if not os.path.exists(file):
        print("\nERROR: Required file not found:")
        print(file)
        raise FileNotFoundError(file)

print("All required files found successfully.")

# ============================================================
# LOAD INDIVIDUAL MODEL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("LOADING INDIVIDUAL ML RESULTS")
print("=" * 70)

individual_results = pd.read_csv(individual_results_file)
individual_timing = pd.read_csv(individual_timing_file)

print("\nIndividual model results:")
print(individual_results)

print("\nIndividual model timing:")
print(individual_timing)

# ============================================================
# LOAD HYBRID RESULTS
# ============================================================

print("\n" + "=" * 70)
print("LOADING HYBRID RESULTS")
print("=" * 70)

hybrid_results = pd.read_csv(hybrid_results_file)
hybrid_timing = pd.read_csv(hybrid_timing_file)

print("\nHybrid results:")
print(hybrid_results)

print("\nHybrid timing:")
print(hybrid_timing)

# ============================================================
# LOAD MLP RESULTS
# ============================================================

print("\n" + "=" * 70)
print("LOADING LIGHTWEIGHT MLP RESULTS")
print("=" * 70)

mlp_results = pd.read_csv(mlp_results_file)

print("\nMLP results:")
print(mlp_results)

# ============================================================
# DISPLAY COLUMN NAMES
# ============================================================

print("\n" + "=" * 70)
print("AVAILABLE COLUMNS")
print("=" * 70)

print("\nIndividual results columns:")
print(individual_results.columns.tolist())

print("\nIndividual timing columns:")
print(individual_timing.columns.tolist())

print("\nHybrid results columns:")
print(hybrid_results.columns.tolist())

print("\nHybrid timing columns:")
print(hybrid_timing.columns.tolist())

print("\nMLP results columns:")
print(mlp_results.columns.tolist())

# ============================================================
# HELPER FUNCTION
# ============================================================


def find_column(df, possible_names):

    for name in possible_names:

        if name in df.columns:
            return name

    return None


# ============================================================
# INDIVIDUAL ML MODELS
# ============================================================

individual_table = individual_results.copy()

# Rename common columns if necessary

rename_map = {}

for column in individual_table.columns:

    lower = column.lower()

    if lower == "model":
        rename_map[column] = "Model"

    elif lower == "accuracy":
        rename_map[column] = "Accuracy"

    elif lower == "precision":
        rename_map[column] = "Precision"

    elif lower == "recall":
        rename_map[column] = "Recall"

    elif lower in ["f1", "f1_score", "f1score"]:
        rename_map[column] = "F1_Score"

individual_table.rename(columns=rename_map, inplace=True)

# ============================================================
# ADD TIMING INFORMATION
# ============================================================

if "Model" in individual_timing.columns:

    timing = individual_timing.copy()

    timing_rename = {}

    for column in timing.columns:

        lower = column.lower()

        if lower == "model":
            timing_rename[column] = "Model"

        elif "training_time_seconds" in lower:
            timing_rename[column] = "Training_Time_Seconds"

        elif "inference_time_seconds" in lower:
            timing_rename[column] = "Inference_Time_Seconds"

        elif "latency_per_sample" in lower:
            timing_rename[column] = "Latency_Per_Sample_ms"

    timing.rename(columns=timing_rename, inplace=True)

    merge_columns = [
        "Model",
        "Training_Time_Seconds",
        "Inference_Time_Seconds",
        "Latency_Per_Sample_ms"
    ]

    merge_columns = [
        c for c in merge_columns
        if c in timing.columns
    ]

    individual_table = individual_table.merge(
        timing[merge_columns],
        on="Model",
        how="left"
    )

# ============================================================
# KEEP MAIN ML MODELS
# ============================================================

ml_models = individual_table.copy()

# ============================================================
# HYBRID MODEL
# ============================================================

hybrid_row = {}

# Results

if len(hybrid_results) > 0:

    row = hybrid_results.iloc[0]

    for column in hybrid_results.columns:

        lower = column.lower()

        if lower == "accuracy":
            hybrid_row["Accuracy"] = row[column]

        elif lower == "precision":
            hybrid_row["Precision"] = row[column]

        elif lower == "recall":
            hybrid_row["Recall"] = row[column]

        elif lower in ["f1", "f1_score", "f1score"]:
            hybrid_row["F1_Score"] = row[column]


# Timing

if len(hybrid_timing) > 0:

    row = hybrid_timing.iloc[0]

    for column in hybrid_timing.columns:

        lower = column.lower()

        if "training_time_seconds" in lower:
            hybrid_row["Training_Time_Seconds"] = row[column]

        elif "inference_time_seconds" in lower:
            hybrid_row["Inference_Time_Seconds"] = row[column]

        elif "latency_per_sample" in lower:
            hybrid_row["Latency_Per_Sample_ms"] = row[column]

        elif "model_size" in lower:
            hybrid_row["Model_Size_MB"] = row[column]


hybrid_row["Model"] = "Four-Classifier Hybrid"

hybrid_table = pd.DataFrame([hybrid_row])

# ============================================================
# MLP MODEL
# ============================================================

mlp_row = {}

if len(mlp_results) > 0:

    row = mlp_results.iloc[0]

    for column in mlp_results.columns:

        lower = column.lower()

        if lower == "accuracy":
            mlp_row["Accuracy"] = row[column]

        elif lower == "precision":
            mlp_row["Precision"] = row[column]

        elif lower == "recall":
            mlp_row["Recall"] = row[column]

        elif lower in ["f1", "f1_score", "f1score"]:
            mlp_row["F1_Score"] = row[column]

        elif "training_time" in lower:
            mlp_row["Training_Time_Seconds"] = row[column]

        elif "inference_time" in lower:
            mlp_row["Inference_Time_Seconds"] = row[column]

        elif "latency_per_sample" in lower:
            mlp_row["Latency_Per_Sample_ms"] = row[column]

        elif "model_size" in lower:
            mlp_row["Model_Size_MB"] = row[column]

        elif "parameter" in lower:
            mlp_row["Parameters"] = row[column]

mlp_row["Model"] = "Lightweight MLP"

mlp_table = pd.DataFrame([mlp_row])

# ============================================================
# COMBINE HYBRID + MLP
# ============================================================

print("\n" + "=" * 70)
print("CREATING ML VS DL COMPARISON")
print("=" * 70)

comparison = pd.concat(
    [
        ml_models,
        hybrid_table,
        mlp_table
    ],
    ignore_index=True
)

# ============================================================
# ORDER COLUMNS
# ============================================================

preferred_columns = [
    "Model",
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score",
    "Training_Time_Seconds",
    "Inference_Time_Seconds",
    "Latency_Per_Sample_ms",
    "Model_Size_MB",
    "Parameters"
]

available_columns = [
    column
    for column in preferred_columns
    if column in comparison.columns
]

comparison = comparison[available_columns]

# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\nFINAL ML VS DL COMPARISON")
print("=" * 70)

print(
    comparison.to_string(
        index=False
    )
)

# ============================================================
# SAVE FULL COMPARISON
# ============================================================

comparison_file = os.path.join(
    RESULTS_DIR,
    "ml_vs_dl_comparison.csv"
)

comparison.to_csv(
    comparison_file,
    index=False
)

print("\nSaved:")
print(comparison_file)

# ============================================================
# PAPER TABLE
# ============================================================

paper_table = comparison.copy()

# Convert metrics to percentage

for column in [
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score"
]:

    if column in paper_table.columns:

        paper_table[column] = (
            paper_table[column] * 100
        ).round(4)

# Round computational values

for column in [
    "Training_Time_Seconds",
    "Inference_Time_Seconds",
    "Latency_Per_Sample_ms",
    "Model_Size_MB"
]:

    if column in paper_table.columns:

        paper_table[column] = paper_table[column].round(6)

paper_table_file = os.path.join(
    RESULTS_DIR,
    "ml_vs_dl_comparison_paper_table.csv"
)

paper_table.to_csv(
    paper_table_file,
    index=False
)

print("\nPaper table saved:")
print(paper_table_file)

# ============================================================
# FIGURE 1 - ACCURACY
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    comparison["Model"],
    comparison["Accuracy"] * 100
)

plt.ylabel("Accuracy (%)")
plt.xlabel("Model")
plt.title("ML and Lightweight DL Accuracy Comparison")

plt.xticks(
    rotation=25,
    ha="right"
)

plt.tight_layout()

accuracy_file = os.path.join(
    FIGURES_DIR,
    "ml_vs_dl_accuracy.png"
)

plt.savefig(
    accuracy_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:", accuracy_file)

# ============================================================
# FIGURE 2 - F1 SCORE
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    comparison["Model"],
    comparison["F1_Score"] * 100
)

plt.ylabel("F1 Score (%)")
plt.xlabel("Model")
plt.title("ML and Lightweight DL F1 Score Comparison")

plt.xticks(
    rotation=25,
    ha="right"
)

plt.tight_layout()

f1_file = os.path.join(
    FIGURES_DIR,
    "ml_vs_dl_f1_score.png"
)

plt.savefig(
    f1_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:", f1_file)

# ============================================================
# FIGURE 3 - TRAINING TIME
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    comparison["Model"],
    comparison["Training_Time_Seconds"]
)

plt.ylabel("Training Time (seconds)")
plt.xlabel("Model")
plt.title("Training Time Comparison")

plt.xticks(
    rotation=25,
    ha="right"
)

plt.tight_layout()

training_file = os.path.join(
    FIGURES_DIR,
    "ml_vs_dl_training_time.png"
)

plt.savefig(
    training_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:", training_file)

# ============================================================
# FIGURE 4 - INFERENCE TIME
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    comparison["Model"],
    comparison["Inference_Time_Seconds"]
)

plt.ylabel("Inference Time (seconds)")
plt.xlabel("Model")
plt.title("Inference Time Comparison")

plt.xticks(
    rotation=25,
    ha="right"
)

plt.tight_layout()

inference_file = os.path.join(
    FIGURES_DIR,
    "ml_vs_dl_inference_time.png"
)

plt.savefig(
    inference_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:", inference_file)

# ============================================================
# FIGURE 5 - MODEL SIZE
# ============================================================

if "Model_Size_MB" in comparison.columns:

    plt.figure(figsize=(10, 6))

    plt.bar(
        comparison["Model"],
        comparison["Model_Size_MB"].fillna(0)
    )

    plt.ylabel("Model Size (MB)")
    plt.xlabel("Model")
    plt.title("Model Size Comparison")

    plt.xticks(
        rotation=25,
        ha="right"
    )

    plt.tight_layout()

    size_file = os.path.join(
        FIGURES_DIR,
        "ml_vs_dl_model_size.png"
    )

    plt.savefig(
        size_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", size_file)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 11A COMPLETED")
print("=" * 70)

print("\nGenerated files:")
print("- ml_vs_dl_comparison.csv")
print("- ml_vs_dl_comparison_paper_table.csv")
print("- figures/ml_vs_dl_accuracy.png")
print("- figures/ml_vs_dl_f1_score.png")
print("- figures/ml_vs_dl_training_time.png")
print("- figures/ml_vs_dl_inference_time.png")
print("- figures/ml_vs_dl_model_size.png")

print("\nNo models were retrained.")
print("Existing ML and MLP results were reused.")

print("\nNext step:")
print("Direct comparison with cited DRX and LR+NB+DT systems.")
print("=" * 70)
