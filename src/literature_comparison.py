"""
STEP 11B - LITERATURE-BASED COMPARISON

Purpose:
    Compare the proposed CICIDS2017 four-classifier soft-voting
    ensemble with selected published IDS studies.

Important:
    The literature values are entered only from the literature
    summary supplied for this research project.

    Results from different datasets, classification settings,
    and metrics should NOT be interpreted as directly equivalent.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


# ==============================================================
# PATH CONFIGURATION
# ==============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ==============================================================
# STEP 11B
# ==============================================================

print("=" * 70)
print("STEP 11B - LITERATURE-BASED COMPARISON")
print("=" * 70)


# ==============================================================
# LOAD PROPOSED MODEL RESULT
# ==============================================================

hybrid_file = os.path.join(
    RESULTS_DIR,
    "hybrid_model_results.csv"
)

if not os.path.exists(hybrid_file):
    raise FileNotFoundError(
        f"Required file not found:\n{hybrid_file}"
    )

print("\nLoading proposed hybrid results...")

hybrid_df = pd.read_csv(hybrid_file)

print("\nProposed hybrid result:")
print(hybrid_df.to_string(index=False))


# ==============================================================
# EXTRACT PROPOSED MODEL VALUES
# ==============================================================

hybrid = hybrid_df.iloc[0]

proposed_accuracy = hybrid["Accuracy"] * 100
proposed_precision = hybrid["Precision"] * 100
proposed_recall = hybrid["Recall"] * 100
proposed_f1 = hybrid["F1_Score"] * 100


# ==============================================================
# LITERATURE DATA
# ==============================================================

literature_data = [

    {
        "Study": "Proposed Method",
        "Dataset": "CICIDS2017",
        "Approach": "LR + DT + RF + XGBoost Soft Voting",
        "Classification": "15-class",
        "Metric": "Accuracy",
        "Accuracy_Percent": proposed_accuracy,
        "Precision_Percent": proposed_precision,
        "Recall_Percent": proposed_recall,
        "F1_Percent": proposed_f1,
        "Direct_Comparison": "Yes"
    },

    {
        "Study": "Hizal et al.",
        "Dataset": "NSL-KDD",
        "Approach": "Conv1D + GRU",
        "Classification": "Binary",
        "Metric": "Accuracy",
        "Accuracy_Percent": 99.88,
        "Precision_Percent": None,
        "Recall_Percent": None,
        "F1_Percent": None,
        "Direct_Comparison": "No"
    },

    {
        "Study": "Hizal et al.",
        "Dataset": "NSL-KDD",
        "Approach": "Conv1D + GRU",
        "Classification": "Five-class",
        "Metric": "Accuracy",
        "Accuracy_Percent": 99.86,
        "Precision_Percent": None,
        "Recall_Percent": None,
        "F1_Percent": None,
        "Direct_Comparison": "No"
    },

    {
        "Study": "Aburomman et al.",
        "Dataset": "KDD99",
        "Approach": "SVM-kNN-PSO / LUS-PSO",
        "Classification": "Ensemble",
        "Metric": "Average Accuracy",
        "Accuracy_Percent": 92.90,
        "Precision_Percent": None,
        "Recall_Percent": None,
        "F1_Percent": None,
        "Direct_Comparison": "No"
    },

    {
        "Study": "Ambusaidi et al.",
        "Dataset": "KDD Cup 99",
        "Approach": "FMIFS + LSSVM",
        "Classification": "IDS classification",
        "Metric": "Accuracy",
        "Accuracy_Percent": 99.79,
        "Precision_Percent": None,
        "Recall_Percent": None,
        "F1_Percent": None,
        "Direct_Comparison": "No"
    },

    {
        "Study": "Ambusaidi et al.",
        "Dataset": "NSL-KDD",
        "Approach": "FMIFS + LSSVM",
        "Classification": "IDS classification",
        "Metric": "Accuracy",
        "Accuracy_Percent": 99.91,
        "Precision_Percent": None,
        "Recall_Percent": None,
        "F1_Percent": None,
        "Direct_Comparison": "No"
    },

    {
        "Study": "Ambusaidi et al.",
        "Dataset": "Kyoto 2006+",
        "Approach": "FMIFS + LSSVM",
        "Classification": "IDS classification",
        "Metric": "Accuracy",
        "Accuracy_Percent": 99.77,
        "Precision_Percent": None,
        "Recall_Percent": None,
        "F1_Percent": None,
        "Direct_Comparison": "No"
    },

    {
        "Study": "Qazi et al.",
        "Dataset": "CICIDS2018",
        "Approach": "CNN + Deep RNN (HDLNIDS)",
        "Classification": "Attack detection",
        "Metric": "Average Accuracy",
        "Accuracy_Percent": 98.90,
        "Precision_Percent": None,
        "Recall_Percent": None,
        "F1_Percent": None,
        "Direct_Comparison": "No"
    },

    {
        "Study": "Verma & Ranga",
        "Dataset": "CIDDS-001 / UNSW-NB15 / NSL-KDD",
        "Approach": "CART, RF, XGB and ensemble ML",
        "Classification": "Supervised IDS",
        "Metric": "Sensitivity",
        "Accuracy_Percent": 97.31,
        "Precision_Percent": None,
        "Recall_Percent": None,
        "F1_Percent": None,
        "Direct_Comparison": "No"
    },

    {
        "Study": "Al Lail et al.",
        "Dataset": "CICIDS2017",
        "Approach": "RF + SMOTE",
        "Classification": "Attack detection",
        "Metric": "Macro F1",
        "Accuracy_Percent": 97.00,
        "Precision_Percent": None,
        "Recall_Percent": None,
        "F1_Percent": 97.00,
        "Direct_Comparison": "No"
    },

    {
        "Study": "Khan et al.",
        "Dataset": "ToN-IoT",
        "Approach": "RNN-GRU",
        "Classification": "Binary network-flow",
        "Metric": "Accuracy",
        "Accuracy_Percent": 99.00,
        "Precision_Percent": None,
        "Recall_Percent": None,
        "F1_Percent": None,
        "Direct_Comparison": "No"
    },

]


# ==============================================================
# CREATE DATAFRAME
# ==============================================================

comparison_df = pd.DataFrame(literature_data)

print("\n" + "=" * 70)
print("LITERATURE COMPARISON")
print("=" * 70)

print(
    comparison_df[
        [
            "Study",
            "Dataset",
            "Approach",
            "Classification",
            "Metric",
            "Accuracy_Percent"
        ]
    ].to_string(index=False)
)


# ==============================================================
# SAVE COMPLETE COMPARISON
# ==============================================================

comparison_file = os.path.join(
    RESULTS_DIR,
    "literature_comparison.csv"
)

comparison_df.to_csv(
    comparison_file,
    index=False
)

print("\nSaved:")
print(comparison_file)


# ==============================================================
# PAPER TABLE
# ==============================================================

paper_table = comparison_df[
    [
        "Study",
        "Dataset",
        "Approach",
        "Classification",
        "Metric",
        "Accuracy_Percent",
        "Direct_Comparison"
    ]
].copy()


paper_table = paper_table.rename(
    columns={
        "Study": "Paper / Model",
        "Dataset": "Dataset",
        "Approach": "Proposed Approach",
        "Classification": "Classification Setting",
        "Metric": "Reported Metric",
        "Accuracy_Percent": "Reported Performance (%)",
        "Direct_Comparison": "Direct Comparison"
    }
)


paper_table_file = os.path.join(
    RESULTS_DIR,
    "literature_comparison_paper_table.csv"
)

paper_table.to_csv(
    paper_table_file,
    index=False
)

print("\nPaper table saved:")
print(paper_table_file)


# ==============================================================
# FIGURE 1 - LITERATURE PERFORMANCE COMPARISON
# ==============================================================

print("\nCreating literature comparison figure...")

plot_df = comparison_df.copy()

# Short labels for the figure
plot_df["Label"] = (
    plot_df["Study"]
    + " - "
    + plot_df["Dataset"]
)

plt.figure(figsize=(13, 7))

bars = plt.bar(
    range(len(plot_df)),
    plot_df["Accuracy_Percent"]
)

plt.xticks(
    range(len(plot_df)),
    plot_df["Label"],
    rotation=55,
    ha="right"
)

plt.ylabel("Reported Performance (%)")
plt.xlabel("Study / Dataset")
plt.title(
    "Performance Comparison with Selected Literature"
)

plt.ylim(
    min(plot_df["Accuracy_Percent"]) - 2,
    100.5
)

# Add values above bars
for i, value in enumerate(
    plot_df["Accuracy_Percent"]
):
    plt.text(
        i,
        value + 0.08,
        f"{value:.2f}%",
        ha="center",
        va="bottom",
        fontsize=8
    )

plt.tight_layout()

figure_file = os.path.join(
    FIGURES_DIR,
    "literature_accuracy_comparison.png"
)

plt.savefig(
    figure_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:")
print(figure_file)


# ==============================================================
# PROPOSED MODEL ANALYSIS
# ==============================================================

print("\n" + "=" * 70)
print("PROPOSED MODEL ANALYSIS")
print("=" * 70)

print(
    f"\nProposed Hybrid Accuracy : "
    f"{proposed_accuracy:.4f}%"
)

print(
    f"Proposed Hybrid Precision: "
    f"{proposed_precision:.4f}%"
)

print(
    f"Proposed Hybrid Recall   : "
    f"{proposed_recall:.4f}%"
)

print(
    f"Proposed Hybrid F1       : "
    f"{proposed_f1:.4f}%"
)


# ==============================================================
# IMPORTANT COMPARISON NOTE
# ==============================================================

print("\n" + "=" * 70)
print("COMPARISON NOTE")
print("=" * 70)

print(
    """
The literature results should not be interpreted as direct
one-to-one comparisons because the studies use different:

- datasets
- class distributions
- classification settings
- preprocessing methods
- evaluation protocols
- reported performance metrics

The proposed method uses 15-class CICIDS2017 classification.
Therefore, binary accuracy, five-class accuracy, sensitivity,
macro-F1, and multiclass accuracy from other studies should
be discussed separately.
"""
)


# ==============================================================
# FINAL SUMMARY
# ==============================================================

print("\n" + "=" * 70)
print("STEP 11B COMPLETED")
print("=" * 70)

print("\nGenerated files:")

print("- literature_comparison.csv")
print("- literature_comparison_paper_table.csv")
print("- figures/literature_accuracy_comparison.png")

print("\nNext step:")
print("Final performance analysis and manuscript-ready discussion.")
