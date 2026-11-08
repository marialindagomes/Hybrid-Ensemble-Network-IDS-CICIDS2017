# ==========================================
# STEP 9
# ABLATION STUDY
# Hybrid ML IDS using CICIDS2017
# ==========================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import time
import warnings

warnings.filterwarnings("ignore")


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DATA_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

MODELS_DIR = (
    PROJECT_ROOT / "models"
)

RESULTS_DIR = (
    PROJECT_ROOT / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================
# 2. INPUT FILES
# ==========================================

X_TEST_FILE = (
    PROCESSED_DATA_DIR /
    "X_test_scaled.csv"
)

Y_TEST_FILE = (
    PROCESSED_DATA_DIR /
    "y_test.csv"
)

LABEL_ENCODER_FILE = (
    PROCESSED_DATA_DIR /
    "label_encoder.pkl"
)


MODEL_FILES = {

    "LR":
        MODELS_DIR /
        "logistic_regression.pkl",

    "DT":
        MODELS_DIR /
        "decision_tree.pkl",

    "RF":
        MODELS_DIR /
        "random_forest.pkl",

    "XGB":
        MODELS_DIR /
        "xgboost.pkl"
}


# ==========================================
# 3. OUTPUT FILE
# ==========================================

ABLATION_RESULTS_FILE = (
    RESULTS_DIR /
    "ablation_study_results.csv"
)


ABLATION_TIMING_FILE = (
    RESULTS_DIR /
    "ablation_study_timing.csv"
)


# ==========================================
# 4. START
# ==========================================

print("=" * 70)
print("STEP 9 - ABLATION STUDY")
print("=" * 70)

total_start = time.perf_counter()


# ==========================================
# 5. CHECK FILES
# ==========================================

print("\nChecking required files...")

required_files = [

    X_TEST_FILE,
    Y_TEST_FILE,
    LABEL_ENCODER_FILE

]

required_files.extend(
    MODEL_FILES.values()
)


for file in required_files:

    if not file.exists():

        print("\nERROR: File not found:")
        print(file)

        print(
            "\nPlease make sure all previous steps "
            "are completed."
        )

        exit()


print(
    "All required files found successfully."
)


# ==========================================
# 6. LOAD TEST DATA
# ==========================================

print("\n" + "=" * 70)
print("LOADING TEST DATA")
print("=" * 70)


print("\nLoading X_test...")

X_test = pd.read_csv(
    X_TEST_FILE,
    low_memory=False
)

print(
    "X_test shape:",
    X_test.shape
)


print("\nLoading y_test...")

y_test = pd.read_csv(
    Y_TEST_FILE
)["Label"]

print(
    "y_test shape:",
    y_test.shape
)


y_test = y_test.astype(
    np.int32
)


# ==========================================
# 7. LOAD LABEL ENCODER
# ==========================================

label_encoder = joblib.load(
    LABEL_ENCODER_FILE
)

class_names = label_encoder.classes_


print("\nNumber of classes:")

print(
    len(class_names)
)


# ==========================================
# 8. LOAD FOUR TRAINED MODELS
# ==========================================

print("\n" + "=" * 70)
print("LOADING TRAINED CLASSIFIERS")
print("=" * 70)


models = {}


for name, file in MODEL_FILES.items():

    print(
        f"\nLoading {name}..."
    )

    models[name] = joblib.load(
        file
    )

    print(
        f"{name} loaded successfully."
    )


# ==========================================
# 9. GENERATE PROBABILITY PREDICTIONS
# ==========================================

print("\n" + "=" * 70)
print("GENERATING PROBABILITY PREDICTIONS")
print("=" * 70)


probabilities = {}


for name, model in models.items():

    print(
        f"\nGenerating probabilities: {name}"
    )

    start = time.perf_counter()

    probabilities[name] = (
        model.predict_proba(X_test)
    )

    end = time.perf_counter()

    elapsed = end - start

    print(
        f"Time: {elapsed:.4f} seconds"
    )

    print(
        "Probability shape:",
        probabilities[name].shape
    )


# ==========================================
# 10. DEFINE ABLATION COMBINATIONS
# ==========================================

combinations = {

    # -------------------------
    # Individual
    # -------------------------

    "LR": [
        "LR"
    ],

    "DT": [
        "DT"
    ],

    "RF": [
        "RF"
    ],

    "XGB": [
        "XGB"
    ],


    # -------------------------
    # Two-model combinations
    # -------------------------

    "LR + DT": [
        "LR",
        "DT"
    ],

    "LR + RF": [
        "LR",
        "RF"
    ],

    "LR + XGB": [
        "LR",
        "XGB"
    ],

    "DT + RF": [
        "DT",
        "RF"
    ],

    "DT + XGB": [
        "DT",
        "XGB"
    ],

    "RF + XGB": [
        "RF",
        "XGB"
    ],


    # -------------------------
    # Three-model combinations
    # -------------------------

    "LR + DT + RF": [
        "LR",
        "DT",
        "RF"
    ],

    "LR + DT + XGB": [
        "LR",
        "DT",
        "XGB"
    ],

    "LR + RF + XGB": [
        "LR",
        "RF",
        "XGB"
    ],

    "DT + RF + XGB": [
        "DT",
        "RF",
        "XGB"
    ],


    # -------------------------
    # Four-model hybrid
    # -------------------------

    "LR + DT + RF + XGB": [
        "LR",
        "DT",
        "RF",
        "XGB"
    ]

}


# ==========================================
# 11. ABLATION EXPERIMENT
# ==========================================

print("\n" + "=" * 70)
print("RUNNING ABLATION EXPERIMENT")
print("=" * 70)


results = []


for combination_name, model_names in combinations.items():

    print("\n" + "-" * 70)

    print(
        "Combination:",
        combination_name
    )

    print(
        "Models:",
        ", ".join(model_names)
    )

    # --------------------------------------
    # Start timing
    # --------------------------------------

    inference_start = time.perf_counter()

    # --------------------------------------
    # Average probability predictions
    # --------------------------------------

    combined_probability = np.mean(

        [
            probabilities[name]
            for name in model_names
        ],

        axis=0

    )

    # --------------------------------------
    # Final prediction
    # --------------------------------------

    prediction = np.argmax(
        combined_probability,
        axis=1
    )

    # --------------------------------------
    # End timing
    # --------------------------------------

    inference_end = time.perf_counter()

    inference_time = (
        inference_end -
        inference_start
    )

    # --------------------------------------
    # Metrics
    # --------------------------------------

    accuracy = accuracy_score(
        y_test,
        prediction
    )

    precision = precision_score(
        y_test,
        prediction,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        prediction,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        prediction,
        average="weighted",
        zero_division=0
    )

    # --------------------------------------
    # Latency
    # --------------------------------------

    test_samples = len(
        X_test
    )

    latency_ms = (
        inference_time /
        test_samples
    ) * 1000

    # --------------------------------------
    # Display
    # --------------------------------------

    print(
        f"Accuracy  : {accuracy:.6f}"
    )

    print(
        f"Precision : {precision:.6f}"
    )

    print(
        f"Recall    : {recall:.6f}"
    )

    print(
        f"F1 Score  : {f1:.6f}"
    )

    print(
        f"Inference : {inference_time:.6f} sec"
    )

    print(
        f"Latency   : {latency_ms:.6f} ms/sample"
    )

    # --------------------------------------
    # Save result
    # --------------------------------------

    results.append({

        "Combination":
            combination_name,

        "Number_of_Classifiers":
            len(model_names),

        "Accuracy":
            accuracy,

        "Precision":
            precision,

        "Recall":
            recall,

        "F1_Score":
            f1,

        "Inference_Time_Seconds":
            inference_time,

        "Latency_Per_Sample_ms":
            latency_ms

    })


# ==========================================
# 12. CREATE RESULTS DATAFRAME
# ==========================================

results_df = pd.DataFrame(
    results
)


# ==========================================
# 13. SAVE RESULTS
# ==========================================

results_df.to_csv(
    ABLATION_RESULTS_FILE,
    index=False
)


# ==========================================
# 14. DISPLAY FINAL TABLE
# ==========================================

print("\n" + "=" * 70)
print("FINAL ABLATION RESULTS")
print("=" * 70)


print(
    results_df.to_string(
        index=False
    )
)


# ==========================================
# 15. BEST CONFIGURATION
# ==========================================

best_accuracy_row = (
    results_df
    .loc[
        results_df["Accuracy"].idxmax()
    ]
)


best_f1_row = (
    results_df
    .loc[
        results_df["F1_Score"].idxmax()
    ]
)


print("\n" + "=" * 70)
print("BEST CONFIGURATIONS")
print("=" * 70)


print(
    "\nHighest Accuracy:"
)

print(
    best_accuracy_row[
        [
            "Combination",
            "Accuracy",
            "F1_Score"
        ]
    ].to_string()
)


print(
    "\nHighest F1 Score:"
)

print(
    best_f1_row[
        [
            "Combination",
            "Accuracy",
            "F1_Score"
        ]
    ].to_string()
)


# ==========================================
# 16. SAVE TIMING SUMMARY
# ==========================================

timing_df = results_df[
    [
        "Combination",
        "Number_of_Classifiers",
        "Inference_Time_Seconds",
        "Latency_Per_Sample_ms"
    ]
]


timing_df.to_csv(
    ABLATION_TIMING_FILE,
    index=False
)


# ==========================================
# 17. COMPLETION
# ==========================================

total_time = (
    time.perf_counter()
    -
    total_start
)


print("\n" + "=" * 70)
print("STEP 9 COMPLETED")
print("=" * 70)


print(
    f"\nTotal processing time:"
    f" {total_time / 60:.2f} minutes"
)


print("\nGenerated files:")

print(
    "- ablation_study_results.csv"
)

print(
    "- ablation_study_timing.csv"
)


print("\nImportant:")

print(
    "No classifier was retrained."
)

print(
    "The four previously trained models "
    "were reused."
)

print(
    "All combinations use equal-weight "
    "soft voting."
)


print("\nNext step:")

print(
    "Generate ablation tables and figures."
)

print("=" * 70)
