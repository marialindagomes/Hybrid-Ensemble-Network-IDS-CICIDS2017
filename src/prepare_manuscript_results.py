"""
======================================================================
STEP 13 - MANUSCRIPT RESULTS AND DISCUSSION PREPARATION
======================================================================

Purpose:
    Prepare manuscript-ready tables, statistics, figures, and
    discussion points using existing experimental results.

Important:
    - No models are retrained.
    - No datasets are modified.
    - Existing Step 12 results are reused.
======================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =====================================================================
# PROJECT PATHS
# =====================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def check_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found:\n{path}")
    print(f"FOUND: {os.path.basename(path)}")


def first_existing_column(df, candidates):
    """
    Return the first column that exists in the DataFrame.
    Useful for handling columns created with _x / _y after merges.
    """
    for col in candidates:
        if col in df.columns:
            return col

    raise KeyError(
        f"None of the following columns were found:\n{candidates}\n"
        f"Available columns:\n{list(df.columns)}"
    )


def clean_duplicate_columns(df):
    """
    Resolve duplicate merge columns such as:
        Inference_Time_Seconds_x
        Inference_Time_Seconds_y

    Preference:
        1. Base column
        2. _x
        3. _y
    """

    df = df.copy()

    duplicate_pairs = [
        (
            "Inference_Time_Seconds",
            "Inference_Time_Seconds_x",
            "Inference_Time_Seconds_y",
        ),
        (
            "Latency_Per_Sample_ms",
            "Latency_Per_Sample_ms_x",
            "Latency_Per_Sample_ms_y",
        ),
    ]

    for base, x_col, y_col in duplicate_pairs:

        if base in df.columns:
            continue

        if x_col in df.columns:
            df[base] = df[x_col]
        elif y_col in df.columns:
            df[base] = df[y_col]

    return df


def save_figure(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# =====================================================================
# START
# =====================================================================

print_header("STEP 13 - MANUSCRIPT RESULTS AND DISCUSSION PREPARATION")

print("""
This step analyzes existing results only.
No models will be retrained.
No datasets will be modified.
""")


# =====================================================================
# STEP 13.1 - LOAD STEP 12 RESULTS
# =====================================================================

print_header("LOADING STEP 12 RESULTS")

performance_path = os.path.join(
    RESULTS_DIR,
    "final_performance_table.csv"
)

computational_path = os.path.join(
    RESULTS_DIR,
    "final_computational_cost_table.csv"
)

ablation_path = os.path.join(
    RESULTS_DIR,
    "final_ablation_summary.csv"
)

literature_path = os.path.join(
    RESULTS_DIR,
    "final_literature_comparison.csv"
)

for path in [
    performance_path,
    computational_path,
    ablation_path,
    literature_path,
]:
    check_file(path)

print("\nAll Step 12 result files loaded successfully.")


performance = pd.read_csv(performance_path)
computational = pd.read_csv(computational_path)
ablation = pd.read_csv(ablation_path)
literature = pd.read_csv(literature_path)


# =====================================================================
# CLEAN DATA
# =====================================================================

performance = performance.copy()
computational = computational.copy()
ablation = clean_duplicate_columns(ablation)
literature = literature.copy()


# Convert numerical columns where possible
numeric_columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score",
    "Training_Time_Seconds",
    "Training_Time_Minutes",
    "Inference_Time_Seconds",
    "Latency_Per_Sample_ms",
    "Model_Size_MB",
    "Parameters",
]


for df in [
    performance,
    computational,
    ablation,
]:
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )


# =====================================================================
# SHOW COLUMNS
# =====================================================================

print_header("AVAILABLE ABLATION COLUMNS")

print(list(ablation.columns))


# =====================================================================
# STEP 13.2 - IDENTIFY IMPORTANT MODELS
# =====================================================================

print_header("IDENTIFYING IMPORTANT MODELS")


# Proposed hybrid
hybrid_row = performance[
    performance["Model"].astype(str).str.contains(
        "Hybrid",
        case=False,
        na=False
    )
]

if hybrid_row.empty:
    raise ValueError("Proposed hybrid model was not found.")

hybrid_row = hybrid_row.iloc[0]


# Random Forest
rf_row = performance[
    performance["Model"].astype(str).str.contains(
        "Random Forest",
        case=False,
        na=False
    )
]

if rf_row.empty:
    raise ValueError("Random Forest result was not found.")

rf_row = rf_row.iloc[0]


# Lightweight MLP
mlp_row = performance[
    performance["Model"].astype(str).str.contains(
        "MLP",
        case=False,
        na=False
    )
]

if mlp_row.empty:
    raise ValueError("MLP result was not found.")

mlp_row = mlp_row.iloc[0]


print("\nProposed Hybrid:")
print(hybrid_row)

print("\nRandom Forest:")
print(rf_row)

print("\nLightweight MLP:")
print(mlp_row)


# =====================================================================
# STEP 13.3 - MAIN PERFORMANCE STATISTICS
# =====================================================================

print_header("MAIN PERFORMANCE STATISTICS")


hybrid_accuracy = float(hybrid_row["Accuracy"])
hybrid_precision = float(hybrid_row["Precision"])
hybrid_recall = float(hybrid_row["Recall"])
hybrid_f1 = float(hybrid_row["F1_Score"])

rf_accuracy = float(rf_row["Accuracy"])
rf_f1 = float(rf_row["F1_Score"])

mlp_accuracy = float(mlp_row["Accuracy"])
mlp_f1 = float(mlp_row["F1_Score"])


hybrid_vs_rf_accuracy = (
    hybrid_accuracy - rf_accuracy
) * 100

hybrid_vs_rf_f1 = (
    hybrid_f1 - rf_f1
) * 100

hybrid_vs_mlp_accuracy = (
    hybrid_accuracy - mlp_accuracy
) * 100

hybrid_vs_mlp_f1 = (
    hybrid_f1 - mlp_f1
) * 100


print(f"Hybrid Accuracy : {hybrid_accuracy * 100:.4f}%")
print(f"Hybrid Precision: {hybrid_precision * 100:.4f}%")
print(f"Hybrid Recall   : {hybrid_recall * 100:.4f}%")
print(f"Hybrid F1       : {hybrid_f1 * 100:.4f}%")

print(
    f"\nHybrid vs Random Forest accuracy: "
    f"{hybrid_vs_rf_accuracy:.4f} percentage points"
)

print(
    f"Hybrid vs Random Forest F1: "
    f"{hybrid_vs_rf_f1:.4f} percentage points"
)

print(
    f"\nHybrid vs MLP accuracy: "
    f"{hybrid_vs_mlp_accuracy:.4f} percentage points"
)

print(
    f"Hybrid vs MLP F1: "
    f"{hybrid_vs_mlp_f1:.4f} percentage points"
)


# =====================================================================
# STEP 13.4 - COMPUTATIONAL COST
# =====================================================================

print_header("COMPUTATIONAL COST ANALYSIS")


hybrid_cost = computational[
    computational["Model"].astype(str).str.contains(
        "Hybrid",
        case=False,
        na=False
    )
]

mlp_cost = computational[
    computational["Model"].astype(str).str.contains(
        "MLP",
        case=False,
        na=False
    )
]

if hybrid_cost.empty:
    raise ValueError("Hybrid computational result not found.")

if mlp_cost.empty:
    raise ValueError("MLP computational result not found.")

hybrid_cost = hybrid_cost.iloc[0]
mlp_cost = mlp_cost.iloc[0]


hybrid_training = float(
    hybrid_cost["Training_Time_Minutes"]
)

hybrid_inference = float(
    hybrid_cost["Inference_Time_Seconds"]
)

hybrid_latency = float(
    hybrid_cost["Latency_Per_Sample_ms"]
)

hybrid_size = float(
    hybrid_cost["Model_Size_MB"]
)


mlp_training = float(
    mlp_cost["Training_Time_Minutes"]
)

mlp_inference = float(
    mlp_cost["Inference_Time_Seconds"]
)

mlp_latency = float(
    mlp_cost["Latency_Per_Sample_ms"]
)

mlp_size = float(
    mlp_cost["Model_Size_MB"]
)


training_time_ratio = hybrid_training / mlp_training
inference_time_ratio = hybrid_inference / mlp_inference
latency_ratio = hybrid_latency / mlp_latency
model_size_ratio = hybrid_size / mlp_size


print(f"Hybrid training time : {hybrid_training:.4f} minutes")
print(f"MLP training time    : {mlp_training:.4f} minutes")

print(
    f"\nHybrid inference time: "
    f"{hybrid_inference:.4f} seconds"
)

print(
    f"MLP inference time   : "
    f"{mlp_inference:.4f} seconds"
)

print(
    f"\nHybrid latency: "
    f"{hybrid_latency:.6f} ms/sample"
)

print(
    f"MLP latency   : "
    f"{mlp_latency:.6f} ms/sample"
)

print(
    f"\nHybrid model size: "
    f"{hybrid_size:.4f} MB"
)

print(
    f"MLP model size   : "
    f"{mlp_size:.4f} MB"
)

print(
    f"\nHybrid/MLP training-time ratio : "
    f"{training_time_ratio:.2f}x"
)

print(
    f"Hybrid/MLP inference-time ratio: "
    f"{inference_time_ratio:.2f}x"
)

print(
    f"Hybrid/MLP latency ratio       : "
    f"{latency_ratio:.2f}x"
)

print(
    f"Hybrid/MLP model-size ratio    : "
    f"{model_size_ratio:.2f}x"
)


# =====================================================================
# STEP 13.5 - ABLATION ANALYSIS
# =====================================================================

print_header("ABLATION STUDY ANALYSIS")


# IMPORTANT FIX:
# final_ablation_summary.csv contains:
#
# Inference_Time_Seconds_x
# Inference_Time_Seconds_y
#
# The helper function above creates a clean:
#
# Inference_Time_Seconds
#
# column before this section.


inference_column = first_existing_column(
    ablation,
    [
        "Inference_Time_Seconds",
        "Inference_Time_Seconds_x",
        "Inference_Time_Seconds_y",
    ]
)

latency_column = first_existing_column(
    ablation,
    [
        "Latency_Per_Sample_ms",
        "Latency_Per_Sample_ms_x",
        "Latency_Per_Sample_ms_y",
    ]
)


# Highest accuracy
best_accuracy_idx = ablation["Accuracy"].idxmax()
best_accuracy_row = ablation.loc[best_accuracy_idx]


# Highest F1
best_f1_idx = ablation["F1_Score"].idxmax()
best_f1_row = ablation.loc[best_f1_idx]


# Lowest inference time
fastest_idx = ablation[inference_column].idxmin()
fastest_row = ablation.loc[fastest_idx]


# Lowest latency
lowest_latency_idx = ablation[latency_column].idxmin()
lowest_latency_row = ablation.loc[lowest_latency_idx]


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


print("\nFastest Ablation Configuration:")
print(
    f"Combination : "
    f"{fastest_row['Combination']}"
)

print(
    f"Inference   : "
    f"{fastest_row[inference_column]:.6f} seconds"
)

print(
    f"Latency     : "
    f"{fastest_row[latency_column]:.6f} ms/sample"
)


# =====================================================================
# STEP 13.6 - ABLATION MANUSCRIPT TABLE
# =====================================================================

print_header("CREATING MANUSCRIPT ABLATION TABLE")


ablation_paper = ablation[
    [
        "Combination",
        "Number_of_Classifiers",
        "Accuracy",
        "Precision",
        "Recall",
        "F1_Score",
        inference_column,
        latency_column,
    ]
].copy()


ablation_paper.columns = [
    "Classifier Combination",
    "No. of Classifiers",
    "Accuracy (%)",
    "Precision (%)",
    "Recall (%)",
    "F1 Score (%)",
    "Inference Time (s)",
    "Latency (ms/sample)",
]


ablation_paper_path = os.path.join(
    RESULTS_DIR,
    "step13_ablation_manuscript_table.csv"
)

ablation_paper.to_csv(
    ablation_paper_path,
    index=False
)

print(f"Saved: {ablation_paper_path}")


# =====================================================================
# STEP 13.7 - FINAL MANUSCRIPT PERFORMANCE TABLE
# =====================================================================

print_header("CREATING MANUSCRIPT PERFORMANCE TABLE")


manuscript_performance = performance.copy()


manuscript_performance["Accuracy (%)"] = (
    manuscript_performance["Accuracy"] * 100
)

manuscript_performance["Precision (%)"] = (
    manuscript_performance["Precision"] * 100
)

manuscript_performance["Recall (%)"] = (
    manuscript_performance["Recall"] * 100
)

manuscript_performance["F1 Score (%)"] = (
    manuscript_performance["F1_Score"] * 100
)


manuscript_performance = manuscript_performance[
    [
        "Model",
        "Category",
        "Accuracy (%)",
        "Precision (%)",
        "Recall (%)",
        "F1 Score (%)",
    ]
]


manuscript_performance_path = os.path.join(
    RESULTS_DIR,
    "step13_manuscript_performance_table.csv"
)

manuscript_performance.to_csv(
    manuscript_performance_path,
    index=False
)

print(
    f"Saved: {manuscript_performance_path}"
)


# =====================================================================
# STEP 13.8 - COMPUTATIONAL COST TABLE
# =====================================================================

print_header("CREATING MANUSCRIPT COMPUTATIONAL TABLE")


manuscript_computational = computational.copy()


manuscript_computational = manuscript_computational[
    [
        "Model",
        "Training_Time_Minutes",
        "Inference_Time_Seconds",
        "Latency_Per_Sample_ms",
        "Model_Size_MB",
    ]
].copy()


manuscript_computational.columns = [
    "Model",
    "Training Time (min)",
    "Inference Time (s)",
    "Latency (ms/sample)",
    "Model Size (MB)",
]


computational_table_path = os.path.join(
    RESULTS_DIR,
    "step13_manuscript_computational_table.csv"
)

manuscript_computational.to_csv(
    computational_table_path,
    index=False
)

print(
    f"Saved: {computational_table_path}"
)


# =====================================================================
# STEP 13.9 - LITERATURE TABLE
# =====================================================================

print_header("CREATING MANUSCRIPT LITERATURE TABLE")


literature_table = literature.copy()


literature_table_path = os.path.join(
    RESULTS_DIR,
    "step13_manuscript_literature_table.csv"
)

literature_table.to_csv(
    literature_table_path,
    index=False
)

print(
    f"Saved: {literature_table_path}"
)


# =====================================================================
# STEP 13.10 - GENERATE MANUSCRIPT DISCUSSION TEXT
# =====================================================================

print_header("GENERATING MANUSCRIPT-READY DISCUSSION")


discussion_text = f"""
RESULTS AND DISCUSSION

Overall Performance

The proposed four-classifier soft-voting ensemble, consisting of
Logistic Regression, Decision Tree, Random Forest, and XGBoost,
achieved an accuracy of {hybrid_accuracy * 100:.4f}%,
precision of {hybrid_precision * 100:.4f}%,
recall of {hybrid_recall * 100:.4f}%, and F1 score of
{hybrid_f1 * 100:.4f}% on the untouched CICIDS2017 test set.

Among the individual classifiers, Random Forest achieved an
accuracy of {rf_accuracy * 100:.4f}% and an F1 score of
{rf_f1 * 100:.4f}%. The proposed hybrid therefore provided an
accuracy improvement of {hybrid_vs_rf_accuracy:.4f} percentage
points over Random Forest and an F1 improvement of
{hybrid_vs_rf_f1:.4f} percentage points.

Although the improvement over Random Forest is numerically small,
the ablation analysis provides evidence that classifier
combinations can produce incremental performance gains. In
particular, the RF + XGB combination achieved an accuracy of
{best_f1_row['Accuracy']:.4f}% and the highest F1 score of
{best_f1_row['F1_Score']:.4f}% among the evaluated ablation
configurations.

Ablation Study

The ablation study evaluated all single-classifier, two-classifier,
three-classifier, and four-classifier combinations. The highest
accuracy was obtained by the complete LR + DT + RF + XGB
combination, with an accuracy of
{best_accuracy_row['Accuracy']:.4f}% and an F1 score of
{best_accuracy_row['F1_Score']:.4f}%.

The RF + XGB configuration achieved the highest F1 score,
{best_f1_row['F1_Score']:.4f}%, while using only two classifiers.
This indicates that increasing ensemble size does not necessarily
produce a monotonic improvement in every evaluation metric.
However, the four-classifier configuration achieved the highest
accuracy and was therefore retained as the proposed model.

Comparison with Lightweight Deep Learning Baseline

The lightweight MLP baseline achieved an accuracy of
{mlp_accuracy * 100:.4f}% and an F1 score of
{mlp_f1 * 100:.4f}%.

The proposed hybrid improved accuracy by
{hybrid_vs_mlp_accuracy:.4f} percentage points and F1 score by
{hybrid_vs_mlp_f1:.4f} percentage points compared with the
lightweight MLP.

The MLP required only {mlp_training:.2f} minutes for training,
whereas the proposed hybrid required {hybrid_training:.2f} minutes.
The hybrid therefore required approximately
{training_time_ratio:.2f} times the MLP training time.

For inference, the hybrid required {hybrid_inference:.4f} seconds
for the evaluated test set, compared with {mlp_inference:.4f}
seconds for the MLP. The corresponding per-sample latency was
{hybrid_latency:.6f} ms for the hybrid and
{mlp_latency:.6f} ms for the MLP.

The hybrid model also required substantially more storage,
with a model size of {hybrid_size:.2f} MB compared with
{mlp_size:.4f} MB for the MLP. These results demonstrate a clear
accuracy-computational-cost trade-off between the proposed
ensemble and the lightweight deep-learning baseline.

Computational Cost

The proposed ensemble required {hybrid_training:.2f} minutes
for training and {hybrid_inference:.4f} seconds for inference
on the evaluated test set. Its measured latency was
{hybrid_latency:.6f} ms per sample.

The computational overhead is primarily associated with combining
four classifiers and retaining the trained ensemble components.
Therefore, although the hybrid provides the best overall accuracy
among the evaluated configurations, its computational requirements
should be considered when deployment resources are constrained.

Literature Comparison

The proposed method achieved {hybrid_accuracy * 100:.4f}% accuracy
on the 15-class CICIDS2017 classification task. Results reported
in previous studies were obtained using different datasets,
classification settings, preprocessing procedures, class
distributions, and evaluation protocols.

Consequently, the literature results should not be interpreted as
strict one-to-one comparisons. Instead, they provide contextual
evidence regarding the performance of the proposed approach across
different intrusion-detection settings.

Limitations and Future Work

The experimental results demonstrate strong performance on the
CICIDS2017 benchmark; however, the proposed approach remains
dependent on the characteristics of the evaluated dataset and
preprocessing pipeline.

The computational analysis also shows that the four-classifier
ensemble has greater training time, inference cost, and model
storage requirements than the lightweight MLP baseline.

Future work should therefore investigate computationally efficient
ensemble configurations, evaluation on additional contemporary
network intrusion datasets, real-time traffic deployment, and
robustness against previously unseen or zero-day attacks.
"""


discussion_path = os.path.join(
    RESULTS_DIR,
    "step13_manuscript_results_discussion.txt"
)

with open(
    discussion_path,
    "w",
    encoding="utf-8"
) as f:
    f.write(discussion_text.strip())


print(
    f"Saved: {discussion_path}"
)


# =====================================================================
# STEP 13.11 - GENERATE ABLATION ACCURACY FIGURE
# =====================================================================

print_header("GENERATING MANUSCRIPT FIGURES")


plt.figure(figsize=(12, 7))

plt.bar(
    ablation["Combination"],
    ablation["Accuracy"] * 100
)

plt.xlabel("Classifier Combination")
plt.ylabel("Accuracy (%)")
plt.title("Ablation Study: Accuracy by Classifier Combination")

plt.xticks(
    rotation=60,
    ha="right"
)

ablation_accuracy_fig = os.path.join(
    FIGURES_DIR,
    "step13_ablation_accuracy.png"
)

save_figure(ablation_accuracy_fig)


# =====================================================================
# STEP 13.12 - ABLATION F1 FIGURE
# =====================================================================

plt.figure(figsize=(12, 7))

plt.bar(
    ablation["Combination"],
    ablation["F1_Score"] * 100
)

plt.xlabel("Classifier Combination")
plt.ylabel("F1 Score (%)")
plt.title("Ablation Study: F1 Score by Classifier Combination")

plt.xticks(
    rotation=60,
    ha="right"
)

ablation_f1_fig = os.path.join(
    FIGURES_DIR,
    "step13_ablation_f1_score.png"
)

save_figure(ablation_f1_fig)


# =====================================================================
# STEP 13.13 - MAIN MODEL PERFORMANCE FIGURE
# =====================================================================

plt.figure(figsize=(10, 6))

plt.bar(
    performance["Model"],
    performance["Accuracy"] * 100
)

plt.xlabel("Model")
plt.ylabel("Accuracy (%)")
plt.title("Performance Comparison of Evaluated Models")

plt.xticks(
    rotation=35,
    ha="right"
)

performance_fig = os.path.join(
    FIGURES_DIR,
    "step13_model_accuracy_comparison.png"
)

save_figure(performance_fig)


# =====================================================================
# STEP 13.14 - COMPUTATIONAL TRADE-OFF FIGURE
# =====================================================================

plt.figure(figsize=(10, 6))

plt.bar(
    computational["Model"],
    computational["Training_Time_Minutes"]
)

plt.xlabel("Model")
plt.ylabel("Training Time (minutes)")
plt.title("Training-Time Comparison")

plt.xticks(
    rotation=35,
    ha="right"
)

training_fig = os.path.join(
    FIGURES_DIR,
    "step13_training_time_comparison.png"
)

save_figure(training_fig)


# =====================================================================
# STEP 13.15 - LITERATURE COMPARISON FIGURE
# =====================================================================

if "Study" in literature.columns and "Accuracy_Percent" in literature.columns:

    literature_plot = literature.copy()

    plt.figure(figsize=(12, 7))

    plt.bar(
        literature_plot["Study"],
        literature_plot["Accuracy_Percent"]
    )

    plt.xlabel("Study")
    plt.ylabel("Reported Accuracy (%)")
    plt.title("Literature-Based Accuracy Comparison")

    plt.xticks(
        rotation=60,
        ha="right"
    )

    literature_fig = os.path.join(
        FIGURES_DIR,
        "step13_literature_accuracy_comparison.png"
    )

    save_figure(literature_fig)


# =====================================================================
# STEP 13.16 - FINAL SUMMARY
# =====================================================================

print_header("MANUSCRIPT-READY SUMMARY")


print(
    f"""
Proposed Four-Classifier Hybrid

Accuracy  : {hybrid_accuracy * 100:.4f}%
Precision : {hybrid_precision * 100:.4f}%
Recall    : {hybrid_recall * 100:.4f}%
F1 Score  : {hybrid_f1 * 100:.4f}%


Compared with Random Forest

Accuracy improvement:
{hybrid_vs_rf_accuracy:.4f} percentage points

F1 improvement:
{hybrid_vs_rf_f1:.4f} percentage points


Compared with Lightweight MLP

Accuracy improvement:
{hybrid_vs_mlp_accuracy:.4f} percentage points

F1 improvement:
{hybrid_vs_mlp_f1:.4f} percentage points


Ablation Study

Best Accuracy:
{best_accuracy_row['Combination']}
{best_accuracy_row['Accuracy']:.4f}%

Best F1:
{best_f1_row['Combination']}
{best_f1_row['F1_Score']:.4f}%


Computational Trade-off

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


# =====================================================================
# FINAL VALIDATION
# =====================================================================

print_header("FINAL VALIDATION")


print(
    f"Performance rows : {len(performance)}"
)

print(
    f"Computational rows: {len(computational)}"
)

print(
    f"Ablation rows    : {len(ablation)}"
)

print(
    f"Literature rows  : {len(literature)}"
)


# Check important values
assert len(performance) > 0
assert len(computational) > 0
assert len(ablation) > 0
assert len(literature) > 0

assert hybrid_accuracy > 0
assert hybrid_f1 > 0

print("\nAll validation checks passed.")


# =====================================================================
# COMPLETED
# =====================================================================

print_header("STEP 13 COMPLETED")


print("""
Generated files:

- step13_ablation_manuscript_table.csv
- step13_manuscript_performance_table.csv
- step13_manuscript_computational_table.csv
- step13_manuscript_literature_table.csv
- step13_manuscript_results_discussion.txt

Generated figures:

- figures/step13_ablation_accuracy.png
- figures/step13_ablation_f1_score.png
- figures/step13_model_accuracy_comparison.png
- figures/step13_training_time_comparison.png
- figures/step13_literature_accuracy_comparison.png

No models were retrained.
No datasets were modified.

The Step 13 manuscript-ready results preparation is complete.
""")
