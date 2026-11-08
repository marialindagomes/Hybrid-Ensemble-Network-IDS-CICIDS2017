"""
STEP 12 - FINAL PERFORMANCE ANALYSIS

Purpose:
    Combine all completed experimental results into final
    manuscript-ready performance, computational-cost, ablation,
    and literature-comparison tables and figures.

No models are retrained in this step.

Required previous outputs:
    results/individual_model_results.csv
    results/individual_model_timing.csv
    results/hybrid_model_results.csv
    results/hybrid_model_timing.csv
    results/mlp_model_results.csv
    results/ablation_study_results.csv
    results/ablation_study_timing.csv
    results/literature_comparison.csv

Generated outputs:
    results/final_performance_table.csv
    results/final_computational_cost_table.csv
    results/final_ablation_summary.csv
    results/final_literature_comparison.csv

    results/figures/
        final_accuracy_comparison.png
        final_f1_comparison.png
        final_training_time_comparison.png
        final_inference_latency_comparison.png
        final_model_size_comparison.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


# ==============================================================
# PROJECT PATHS
# ==============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)

FIGURES_DIR = os.path.join(
    RESULTS_DIR,
    "figures"
)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ==============================================================
# HELPER FUNCTIONS
# ==============================================================

def check_file(path):
    """Check whether a required file exists."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}"
        )


def save_csv(df, filename):
    """Save dataframe to results directory."""
    path = os.path.join(
        RESULTS_DIR,
        filename
    )

    df.to_csv(
        path,
        index=False
    )

    print(f"Saved: {path}")


def save_figure(filename):
    """Return figure path."""
    return os.path.join(
        FIGURES_DIR,
        filename
    )


# ==============================================================
# HEADER
# ==============================================================

print("=" * 70)
print("STEP 12 - FINAL PERFORMANCE ANALYSIS")
print("=" * 70)

print("\nThis step only analyzes existing results.")
print("No models will be retrained.")


# ==============================================================
# REQUIRED FILES
# ==============================================================

required_files = [

    "individual_model_results.csv",
    "individual_model_timing.csv",

    "hybrid_model_results.csv",
    "hybrid_model_timing.csv",

    "mlp_model_results.csv",

    "ablation_study_results.csv",
    "ablation_study_timing.csv",

    "literature_comparison.csv"

]


print("\n" + "=" * 70)
print("CHECKING REQUIRED FILES")
print("=" * 70)

for filename in required_files:

    path = os.path.join(
        RESULTS_DIR,
        filename
    )

    check_file(path)

    print(f"FOUND: {filename}")

print("\nAll required files found successfully.")


# ==============================================================
# LOAD INDIVIDUAL MODEL RESULTS
# ==============================================================

print("\n" + "=" * 70)
print("LOADING INDIVIDUAL MODEL RESULTS")
print("=" * 70)

individual_results = pd.read_csv(
    os.path.join(
        RESULTS_DIR,
        "individual_model_results.csv"
    )
)

individual_timing = pd.read_csv(
    os.path.join(
        RESULTS_DIR,
        "individual_model_timing.csv"
    )
)

print("\nIndividual performance:")
print(
    individual_results.to_string(
        index=False
    )
)

print("\nIndividual computational cost:")
print(
    individual_timing.to_string(
        index=False
    )
)


# ==============================================================
# LOAD HYBRID RESULTS
# ==============================================================

print("\n" + "=" * 70)
print("LOADING HYBRID RESULTS")
print("=" * 70)

hybrid_results = pd.read_csv(
    os.path.join(
        RESULTS_DIR,
        "hybrid_model_results.csv"
    )
)

hybrid_timing = pd.read_csv(
    os.path.join(
        RESULTS_DIR,
        "hybrid_model_timing.csv"
    )
)

print("\nHybrid performance:")
print(
    hybrid_results.to_string(
        index=False
    )
)

print("\nHybrid computational cost:")
print(
    hybrid_timing.to_string(
        index=False
    )
)


# ==============================================================
# LOAD MLP RESULTS
# ==============================================================

print("\n" + "=" * 70)
print("LOADING LIGHTWEIGHT MLP RESULTS")
print("=" * 70)

mlp_results = pd.read_csv(
    os.path.join(
        RESULTS_DIR,
        "mlp_model_results.csv"
    )
)

print("\nMLP results:")
print(
    mlp_results.to_string(
        index=False
    )
)


# ==============================================================
# LOAD ABLATION RESULTS
# ==============================================================

print("\n" + "=" * 70)
print("LOADING ABLATION RESULTS")
print("=" * 70)

ablation_results = pd.read_csv(
    os.path.join(
        RESULTS_DIR,
        "ablation_study_results.csv"
    )
)

ablation_timing = pd.read_csv(
    os.path.join(
        RESULTS_DIR,
        "ablation_study_timing.csv"
    )
)

print("\nAblation results:")
print(
    ablation_results.to_string(
        index=False
    )
)


# ==============================================================
# LOAD LITERATURE RESULTS
# ==============================================================

print("\n" + "=" * 70)
print("LOADING LITERATURE COMPARISON")
print("=" * 70)

literature_results = pd.read_csv(
    os.path.join(
        RESULTS_DIR,
        "literature_comparison.csv"
    )
)

print("\nLiterature results:")
print(
    literature_results.to_string(
        index=False
    )
)


# ==============================================================
# CREATE FINAL PERFORMANCE TABLE
# ==============================================================

print("\n" + "=" * 70)
print("CREATING FINAL PERFORMANCE TABLE")
print("=" * 70)


# --------------------------------------------------------------
# Individual ML models
# --------------------------------------------------------------

performance_ml = individual_results[
    [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1_Score"
    ]
].copy()


# --------------------------------------------------------------
# Rename hybrid model
# --------------------------------------------------------------

performance_hybrid = hybrid_results[
    [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1_Score"
    ]
].copy()

performance_hybrid["Model"] = (
    "Four-Classifier Hybrid"
)


# --------------------------------------------------------------
# MLP
# --------------------------------------------------------------

performance_mlp = mlp_results[
    [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1_Score"
    ]
].copy()


# --------------------------------------------------------------
# Combine
# --------------------------------------------------------------

final_performance = pd.concat(
    [
        performance_ml,
        performance_hybrid,
        performance_mlp
    ],
    ignore_index=True
)


# Convert metrics to percentage
metric_columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score"
]

for column in metric_columns:

    final_performance[column] = (
        final_performance[column] * 100
    )


# Add model category

def get_category(model):

    if model == "Four-Classifier Hybrid":
        return "Proposed Ensemble"

    if model == "Lightweight MLP":
        return "Deep Learning Baseline"

    return "Individual ML"


final_performance.insert(
    1,
    "Category",
    final_performance["Model"].apply(
        get_category
    )
)


# Round
final_performance[
    metric_columns
] = final_performance[
    metric_columns
].round(4)


print("\nFINAL PERFORMANCE TABLE")
print("=" * 70)

print(
    final_performance.to_string(
        index=False
    )
)


save_csv(
    final_performance,
    "final_performance_table.csv"
)


# ==============================================================
# FINAL COMPUTATIONAL COST TABLE
# ==============================================================

print("\n" + "=" * 70)
print("CREATING FINAL COMPUTATIONAL COST TABLE")
print("=" * 70)


# --------------------------------------------------------------
# Individual models
# --------------------------------------------------------------

cost_ml = individual_timing.copy()

cost_ml = cost_ml[
    [
        "Model",
        "Training_Time_Seconds",
        "Training_Time_Minutes",
        "Inference_Time_Seconds",
        "Latency_Per_Sample_ms"
    ]
]


# --------------------------------------------------------------
# Hybrid
# --------------------------------------------------------------

cost_hybrid = hybrid_timing.copy()

cost_hybrid["Model"] = (
    "Four-Classifier Hybrid"
)

cost_hybrid = cost_hybrid[
    [
        "Model",
        "Training_Time_Seconds",
        "Training_Time_Minutes",
        "Inference_Time_Seconds",
        "Latency_Per_Sample_ms"
    ]
]


# --------------------------------------------------------------
# MLP
# --------------------------------------------------------------

cost_mlp = mlp_results[
    [
        "Model",
        "Training_Time_Seconds",
        "Training_Time_Minutes",
        "Inference_Time_Seconds",
        "Latency_Per_Sample_ms",
        "Model_Size_MB",
        "Parameters"
    ]
].copy()


# --------------------------------------------------------------
# Add missing columns
# --------------------------------------------------------------

cost_ml["Model_Size_MB"] = None
cost_ml["Parameters"] = None

cost_hybrid["Model_Size_MB"] = (
    hybrid_timing["Model_Size_MB"].iloc[0]
)

cost_hybrid["Parameters"] = None


# --------------------------------------------------------------
# Combine
# --------------------------------------------------------------

final_cost = pd.concat(
    [
        cost_ml,
        cost_hybrid,
        cost_mlp
    ],
    ignore_index=True
)


# Round values

for column in [
    "Training_Time_Seconds",
    "Training_Time_Minutes",
    "Inference_Time_Seconds",
    "Latency_Per_Sample_ms",
    "Model_Size_MB"
]:

    if column in final_cost.columns:

        final_cost[column] = (
            pd.to_numeric(
                final_cost[column],
                errors="coerce"
            ).round(6)
        )


print("\nFINAL COMPUTATIONAL COST TABLE")
print("=" * 70)

print(
    final_cost.to_string(
        index=False
    )
)


save_csv(
    final_cost,
    "final_computational_cost_table.csv"
)


# ==============================================================
# ABLATION SUMMARY
# ==============================================================

print("\n" + "=" * 70)
print("CREATING FINAL ABLATION SUMMARY")
print("=" * 70)


ablation_summary = ablation_results.copy()


# Add classifier count if not present

if "Number_of_Classifiers" not in ablation_summary.columns:

    def count_models(combination):

        return len(
            str(combination).split(" + ")
        )

    ablation_summary[
        "Number_of_Classifiers"
    ] = ablation_summary[
        "Combination"
    ].apply(count_models)


# Convert metrics to percentage

for column in [
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score"
]:

    if column in ablation_summary.columns:

        ablation_summary[column] = (
            ablation_summary[column] * 100
        )


# Add timing if available

timing_columns = [
    "Combination",
    "Inference_Time_Seconds",
    "Latency_Per_Sample_ms"
]

available_timing_columns = [
    column
    for column in timing_columns
    if column in ablation_timing.columns
]

if "Combination" in available_timing_columns:

    ablation_summary = ablation_summary.merge(
        ablation_timing[
            available_timing_columns
        ],
        on="Combination",
        how="left"
    )


# Round

for column in [
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score",
    "Inference_Time_Seconds",
    "Latency_Per_Sample_ms"
]:

    if column in ablation_summary.columns:

        ablation_summary[column] = (
            pd.to_numeric(
                ablation_summary[column],
                errors="coerce"
            ).round(6)
        )


print("\nFINAL ABLATION SUMMARY")
print("=" * 70)

print(
    ablation_summary.to_string(
        index=False
    )
)


save_csv(
    ablation_summary,
    "final_ablation_summary.csv"
)


# ==============================================================
# BEST ABLATION CONFIGURATIONS
# ==============================================================

print("\n" + "=" * 70)
print("BEST ABLATION CONFIGURATIONS")
print("=" * 70)


best_accuracy_row = (
    ablation_summary.loc[
        ablation_summary["Accuracy"].idxmax()
    ]
)

best_f1_row = (
    ablation_summary.loc[
        ablation_summary["F1_Score"].idxmax()
    ]
)


print("\nHighest Accuracy:")
print(
    f"Combination : "
    f"{best_accuracy_row['Combination']}"
)

print(
    f"Accuracy    : "
    f"{best_accuracy_row['Accuracy']:.4f}%"
)

print(
    f"F1 Score    : "
    f"{best_accuracy_row['F1_Score']:.4f}%"
)


print("\nHighest F1 Score:")
print(
    f"Combination : "
    f"{best_f1_row['Combination']}"
)

print(
    f"Accuracy    : "
    f"{best_f1_row['Accuracy']:.4f}%"
)

print(
    f"F1 Score    : "
    f"{best_f1_row['F1_Score']:.4f}%"
)


# ==============================================================
# LITERATURE TABLE
# ==============================================================

print("\n" + "=" * 70)
print("CREATING FINAL LITERATURE COMPARISON TABLE")
print("=" * 70)


final_literature = literature_results.copy()


# Round literature values

if "Accuracy_Percent" in final_literature.columns:

    final_literature[
        "Accuracy_Percent"
    ] = final_literature[
        "Accuracy_Percent"
    ].round(4)


save_csv(
    final_literature,
    "final_literature_comparison.csv"
)


# ==============================================================
# CALCULATE PERFORMANCE DIFFERENCES
# ==============================================================

print("\n" + "=" * 70)
print("CALCULATING PERFORMANCE DIFFERENCES")
print("=" * 70)


hybrid_accuracy = float(
    final_performance.loc[
        final_performance["Model"]
        == "Four-Classifier Hybrid",
        "Accuracy"
    ].iloc[0]
)

hybrid_f1 = float(
    final_performance.loc[
        final_performance["Model"]
        == "Four-Classifier Hybrid",
        "F1_Score"
    ].iloc[0]
)


rf_accuracy = float(
    final_performance.loc[
        final_performance["Model"]
        == "Random Forest",
        "Accuracy"
    ].iloc[0]
)

rf_f1 = float(
    final_performance.loc[
        final_performance["Model"]
        == "Random Forest",
        "F1_Score"
    ].iloc[0]
)


mlp_accuracy = float(
    final_performance.loc[
        final_performance["Model"]
        == "Lightweight MLP",
        "Accuracy"
    ].iloc[0]
)

mlp_f1 = float(
    final_performance.loc[
        final_performance["Model"]
        == "Lightweight MLP",
        "F1_Score"
    ].iloc[0]
)


print(
    f"\nHybrid vs Random Forest accuracy: "
    f"{hybrid_accuracy - rf_accuracy:.4f} percentage points"
)

print(
    f"Hybrid vs Random Forest F1: "
    f"{hybrid_f1 - rf_f1:.4f} percentage points"
)

print(
    f"\nHybrid vs MLP accuracy: "
    f"{hybrid_accuracy - mlp_accuracy:.4f} percentage points"
)

print(
    f"Hybrid vs MLP F1: "
    f"{hybrid_f1 - mlp_f1:.4f} percentage points"
)


# ==============================================================
# COMPUTATIONAL TRADE-OFF
# ==============================================================

print("\n" + "=" * 70)
print("COMPUTATIONAL TRADE-OFF ANALYSIS")
print("=" * 70)


hybrid_training = float(
    final_cost.loc[
        final_cost["Model"]
        == "Four-Classifier Hybrid",
        "Training_Time_Minutes"
    ].iloc[0]
)

hybrid_inference = float(
    final_cost.loc[
        final_cost["Model"]
        == "Four-Classifier Hybrid",
        "Inference_Time_Seconds"
    ].iloc[0]
)

hybrid_latency = float(
    final_cost.loc[
        final_cost["Model"]
        == "Four-Classifier Hybrid",
        "Latency_Per_Sample_ms"
    ].iloc[0]
)

hybrid_size = float(
    final_cost.loc[
        final_cost["Model"]
        == "Four-Classifier Hybrid",
        "Model_Size_MB"
    ].iloc[0]
)


mlp_training = float(
    final_cost.loc[
        final_cost["Model"]
        == "Lightweight MLP",
        "Training_Time_Minutes"
    ].iloc[0]
)

mlp_inference = float(
    final_cost.loc[
        final_cost["Model"]
        == "Lightweight MLP",
        "Inference_Time_Seconds"
    ].iloc[0]
)

mlp_latency = float(
    final_cost.loc[
        final_cost["Model"]
        == "Lightweight MLP",
        "Latency_Per_Sample_ms"
    ].iloc[0]
)

mlp_size = float(
    final_cost.loc[
        final_cost["Model"]
        == "Lightweight MLP",
        "Model_Size_MB"
    ].iloc[0]
)


print(
    f"\nHybrid training time : "
    f"{hybrid_training:.2f} minutes"
)

print(
    f"MLP training time    : "
    f"{mlp_training:.2f} minutes"
)

print(
    f"\nHybrid inference time: "
    f"{hybrid_inference:.4f} seconds"
)

print(
    f"MLP inference time   : "
    f"{mlp_inference:.4f} seconds"
)

print(
    f"\nHybrid latency       : "
    f"{hybrid_latency:.6f} ms/sample"
)

print(
    f"MLP latency          : "
    f"{mlp_latency:.6f} ms/sample"
)

print(
    f"\nHybrid model size    : "
    f"{hybrid_size:.2f} MB"
)

print(
    f"MLP model size       : "
    f"{mlp_size:.4f} MB"
)


# ==============================================================
# FIGURE 1 - ACCURACY
# ==============================================================

print("\n" + "=" * 70)
print("GENERATING FINAL FIGURES")
print("=" * 70)


# --------------------------------------------------------------
# Accuracy
# --------------------------------------------------------------

plt.figure(figsize=(11, 6))

bars = plt.bar(
    final_performance["Model"],
    final_performance["Accuracy"]
)

plt.ylabel("Accuracy (%)")
plt.xlabel("Model")
plt.title(
    "Final Accuracy Comparison"
)

plt.xticks(
    rotation=35,
    ha="right"
)

plt.ylim(
    max(
        0,
        final_performance["Accuracy"].min() - 3
    ),
    100
)

for bar, value in zip(
    bars,
    final_performance["Accuracy"]
):

    plt.text(
        bar.get_x()
        + bar.get_width() / 2,
        value + 0.05,
        f"{value:.2f}%",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.tight_layout()

accuracy_figure = save_figure(
    "final_accuracy_comparison.png"
)

plt.savefig(
    accuracy_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved: {accuracy_figure}"
)


# ==============================================================
# FIGURE 2 - F1 SCORE
# ==============================================================

plt.figure(figsize=(11, 6))

bars = plt.bar(
    final_performance["Model"],
    final_performance["F1_Score"]
)

plt.ylabel("F1 Score (%)")
plt.xlabel("Model")
plt.title(
    "Final F1-Score Comparison"
)

plt.xticks(
    rotation=35,
    ha="right"
)

plt.ylim(
    max(
        0,
        final_performance["F1_Score"].min() - 3
    ),
    100
)

for bar, value in zip(
    bars,
    final_performance["F1_Score"]
):

    plt.text(
        bar.get_x()
        + bar.get_width() / 2,
        value + 0.05,
        f"{value:.2f}%",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.tight_layout()

f1_figure = save_figure(
    "final_f1_comparison.png"
)

plt.savefig(
    f1_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved: {f1_figure}"
)


# ==============================================================
# FIGURE 3 - TRAINING TIME
# ==============================================================

plt.figure(figsize=(11, 6))

bars = plt.bar(
    final_cost["Model"],
    final_cost["Training_Time_Minutes"]
)

plt.ylabel("Training Time (minutes)")
plt.xlabel("Model")
plt.title(
    "Training Time Comparison"
)

plt.xticks(
    rotation=35,
    ha="right"
)

for bar, value in zip(
    bars,
    final_cost["Training_Time_Minutes"]
):

    plt.text(
        bar.get_x()
        + bar.get_width() / 2,
        value + 0.2,
        f"{value:.2f}",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.tight_layout()

training_figure = save_figure(
    "final_training_time_comparison.png"
)

plt.savefig(
    training_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved: {training_figure}"
)


# ==============================================================
# FIGURE 4 - INFERENCE LATENCY
# ==============================================================

plt.figure(figsize=(11, 6))

bars = plt.bar(
    final_cost["Model"],
    final_cost["Latency_Per_Sample_ms"]
)

plt.ylabel(
    "Latency per Sample (ms)"
)

plt.xlabel("Model")

plt.title(
    "Inference Latency Comparison"
)

plt.xticks(
    rotation=35,
    ha="right"
)

for bar, value in zip(
    bars,
    final_cost["Latency_Per_Sample_ms"]
):

    plt.text(
        bar.get_x()
        + bar.get_width() / 2,
        value + 0.0002,
        f"{value:.6f}",
        ha="center",
        va="bottom",
        fontsize=8
    )

plt.tight_layout()

latency_figure = save_figure(
    "final_inference_latency_comparison.png"
)

plt.savefig(
    latency_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved: {latency_figure}"
)


# ==============================================================
# FIGURE 5 - MODEL SIZE
# ==============================================================

size_df = final_cost[
    final_cost["Model_Size_MB"].notna()
].copy()


plt.figure(figsize=(10, 6))

bars = plt.bar(
    size_df["Model"],
    size_df["Model_Size_MB"]
)

plt.ylabel(
    "Model Size (MB)"
)

plt.xlabel("Model")

plt.title(
    "Model Size Comparison"
)

plt.xticks(
    rotation=35,
    ha="right"
)

for bar, value in zip(
    bars,
    size_df["Model_Size_MB"]
):

    plt.text(
        bar.get_x()
        + bar.get_width() / 2,
        value + max(
            size_df["Model_Size_MB"]
        ) * 0.01,
        f"{value:.2f}",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.tight_layout()

size_figure = save_figure(
    "final_model_size_comparison.png"
)

plt.savefig(
    size_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved: {size_figure}"
)


# ==============================================================
# MANUSCRIPT-READY SUMMARY
# ==============================================================

print("\n" + "=" * 70)
print("MANUSCRIPT-READY SUMMARY")
print("=" * 70)

print(
    f"""
The proposed four-classifier soft-voting ensemble achieved:

Accuracy  : {hybrid_accuracy:.4f}%
F1 Score  : {hybrid_f1:.4f}%

Compared with Random Forest:

Accuracy improvement:
{hybrid_accuracy - rf_accuracy:.4f} percentage points

F1 difference:
{hybrid_f1 - rf_f1:.4f} percentage points

Compared with the lightweight MLP:

Accuracy improvement:
{hybrid_accuracy - mlp_accuracy:.4f} percentage points

F1 improvement:
{hybrid_f1 - mlp_f1:.4f} percentage points

Computational trade-off:

Hybrid training time:
{hybrid_training:.2f} minutes

Hybrid inference time:
{hybrid_inference:.4f} seconds

Hybrid latency:
{hybrid_latency:.6f} ms/sample

Hybrid model size:
{hybrid_size:.2f} MB

Lightweight MLP training time:
{mlp_training:.2f} minutes

Lightweight MLP inference time:
{mlp_inference:.4f} seconds

Lightweight MLP latency:
{mlp_latency:.6f} ms/sample

Lightweight MLP model size:
{mlp_size:.4f} MB
"""
)


# ==============================================================
# FINAL VALIDATION
# ==============================================================

print("=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

print(
    f"\nFinal performance rows: "
    f"{len(final_performance)}"
)

print(
    f"Final computational rows: "
    f"{len(final_cost)}"
)

print(
    f"Final ablation rows: "
    f"{len(ablation_summary)}"
)

print(
    f"Literature rows: "
    f"{len(final_literature)}"
)


# ==============================================================
# COMPLETION
# ==============================================================

print("\n" + "=" * 70)
print("STEP 12 COMPLETED")
print("=" * 70)

print("\nGenerated files:")

print("- final_performance_table.csv")
print("- final_computational_cost_table.csv")
print("- final_ablation_summary.csv")
print("- final_literature_comparison.csv")

print("\nGenerated figures:")

print("- figures/final_accuracy_comparison.png")
print("- figures/final_f1_comparison.png")
print("- figures/final_training_time_comparison.png")
print("- figures/final_inference_latency_comparison.png")
print("- figures/final_model_size_comparison.png")

print("\nNo models were retrained.")

print("\nNext step:")
print("Step 13 - Manuscript Results and Discussion Preparation")
