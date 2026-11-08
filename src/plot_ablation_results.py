# ==========================================
# STEP 9B
# ABLATION STUDY - TABLES AND FIGURES
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


# ==========================================
# 1. PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = (
    PROJECT_ROOT / "results"
)

FIGURES_DIR = (
    RESULTS_DIR / "figures"
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


INPUT_FILE = (
    RESULTS_DIR /
    "ablation_study_results.csv"
)


# ==========================================
# 2. LOAD RESULTS
# ==========================================

print("=" * 70)
print("STEP 9B - ABLATION TABLES AND FIGURES")
print("=" * 70)

print("\nLoading ablation results...")

df = pd.read_csv(
    INPUT_FILE
)

print(
    "Rows:",
    len(df)
)


# ==========================================
# 3. CONVERT TO PERCENTAGE
# ==========================================

df["Accuracy_Percent"] = (
    df["Accuracy"] * 100
)

df["Precision_Percent"] = (
    df["Precision"] * 100
)

df["Recall_Percent"] = (
    df["Recall"] * 100
)

df["F1_Percent"] = (
    df["F1_Score"] * 100
)


# ==========================================
# 4. SAVE PAPER TABLE
# ==========================================

paper_table = df[
    [
        "Combination",
        "Number_of_Classifiers",
        "Accuracy_Percent",
        "Precision_Percent",
        "Recall_Percent",
        "F1_Percent",
        "Inference_Time_Seconds",
        "Latency_Per_Sample_ms"
    ]
].copy()


paper_table.columns = [

    "Classifier Combination",
    "No. of Classifiers",
    "Accuracy (%)",
    "Precision (%)",
    "Recall (%)",
    "F1-Score (%)",
    "Inference Time (s)",
    "Latency (ms/sample)"

]


paper_table = paper_table.round(
    {
        "Accuracy (%)": 4,
        "Precision (%)": 4,
        "Recall (%)": 4,
        "F1-Score (%)": 4,
        "Inference Time (s)": 6,
        "Latency (ms/sample)": 6
    }
)


paper_table_file = (
    RESULTS_DIR /
    "ablation_paper_table.csv"
)


paper_table.to_csv(
    paper_table_file,
    index=False
)


print(
    "\nPaper table saved:"
)

print(
    paper_table_file
)


# ==========================================
# 5. FIGURE 1 - ACCURACY
# ==========================================

plt.figure(
    figsize=(14, 7)
)

plt.bar(
    df["Combination"],
    df["Accuracy_Percent"]
)

plt.ylabel(
    "Accuracy (%)"
)

plt.xlabel(
    "Classifier Combination"
)

plt.title(
    "Ablation Study: Accuracy Comparison"
)

plt.xticks(
    rotation=60,
    ha="right"
)

plt.ylim(
    95,
    100
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

accuracy_file = (
    FIGURES_DIR /
    "ablation_accuracy.png"
)

plt.savefig(
    accuracy_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "Saved:",
    accuracy_file
)


# ==========================================
# 6. FIGURE 2 - F1 SCORE
# ==========================================

plt.figure(
    figsize=(14, 7)
)

plt.bar(
    df["Combination"],
    df["F1_Percent"]
)

plt.ylabel(
    "F1-Score (%)"
)

plt.xlabel(
    "Classifier Combination"
)

plt.title(
    "Ablation Study: F1-Score Comparison"
)

plt.xticks(
    rotation=60,
    ha="right"
)

plt.ylim(
    95,
    100
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

f1_file = (
    FIGURES_DIR /
    "ablation_f1_score.png"
)

plt.savefig(
    f1_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "Saved:",
    f1_file
)


# ==========================================
# 7. FIGURE 3
# ACCURACY VS NUMBER OF CLASSIFIERS
# ==========================================

grouped = (
    df
    .groupby(
        "Number_of_Classifiers"
    )["Accuracy_Percent"]
    .max()
)


plt.figure(
    figsize=(9, 6)
)

plt.plot(
    grouped.index,
    grouped.values,
    marker="o"
)

plt.xlabel(
    "Number of Classifiers"
)

plt.ylabel(
    "Best Accuracy (%)"
)

plt.title(
    "Effect of Ensemble Size on Best Accuracy"
)

plt.xticks(
    [1, 2, 3, 4]
)

plt.ylim(
    99.5,
    100
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

ensemble_size_file = (
    FIGURES_DIR /
    "ensemble_size_vs_accuracy.png"
)

plt.savefig(
    ensemble_size_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "Saved:",
    ensemble_size_file
)


# ==========================================
# 8. FIGURE 4
# LATENCY VS NUMBER OF CLASSIFIERS
# ==========================================

grouped_latency = (
    df
    .groupby(
        "Number_of_Classifiers"
    )["Latency_Per_Sample_ms"]
    .mean()
)


plt.figure(
    figsize=(9, 6)
)

plt.plot(
    grouped_latency.index,
    grouped_latency.values,
    marker="o"
)

plt.xlabel(
    "Number of Classifiers"
)

plt.ylabel(
    "Mean Latency (ms/sample)"
)

plt.title(
    "Effect of Ensemble Size on Inference Latency"
)

plt.xticks(
    [1, 2, 3, 4]
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

latency_file = (
    FIGURES_DIR /
    "ensemble_size_vs_latency.png"
)

plt.savefig(
    latency_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "Saved:",
    latency_file
)


# ==========================================
# 9. FINAL
# ==========================================

print("\n" + "=" * 70)
print("STEP 9B COMPLETED")
print("=" * 70)

print("\nGenerated files:")

print(
    "- ablation_paper_table.csv"
)

print(
    "- figures/ablation_accuracy.png"
)

print(
    "- figures/ablation_f1_score.png"
)

print(
    "- figures/ensemble_size_vs_accuracy.png"
)

print(
    "- figures/ensemble_size_vs_latency.png"
)

print("\nThese figures can be used for the revised manuscript.")

print("=" * 70)
