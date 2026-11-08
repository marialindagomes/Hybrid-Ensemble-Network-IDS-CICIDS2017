# 🛡️ An Ensemble Learning-Based Network Intrusion Detection System Using CICIDS2017

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-15B5B0?logo=xgboost&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Conference](https://img.shields.io/badge/ICCIT-2026-purple)

This repository contains the official implementation of the research paper: **"An Ensemble Learning-Based Network Intrusion Detection System Using CICIDS2017"**, presented at the *2026 29th International Conference on Computer and Information Technology (ICCIT)*.

---

## 📖 Overview

With the rapid growth of computer networks and internet-based services, traditional security mechanisms like firewalls and signature-based IDSs struggle to detect modern, evolving cyber threats. This project implements a robust, reproducible **Hybrid Machine Learning-Based Intrusion Detection System (IDS)** using a **Soft Voting Ensemble** approach on the **CICIDS2017** benchmark dataset.

By combining the complementary strengths of **Logistic Regression (LR)**, **Decision Tree (DT)**, **Random Forest (RF)**, and **XGBoost**, our framework achieves **99.64% accuracy, precision, recall, and F1-score**, outperforming individual classifiers and several deep learning baselines while maintaining low computational complexity and false alarm rates.

---

## ✨ Key Features & Contributions

- **Soft Voting Hybrid Ensemble:** Integrates four diverse machine learning classifiers (`LR + DT + RF + XGBoost`) by averaging their predicted class probabilities for superior stability and generalization.
- **Leakage-Free End-to-End Pipeline:** Strict adherence to data integrity—performing **SMOTE** class balancing and **StandardScaler** normalization *only* on the training split after stratified data partitioning (80% Train / 10% Validation / 10% Test).
- **Statistical Feature Selection:** Uses `SelectKBest` with the **ANOVA F-test ($f\_classif$)** to distill 79 raw network flow attributes down to the **top 30 most discriminative features**.
- **Comprehensive Evaluation:** Includes ablation studies, computational cost/timing analysis, ML vs. Deep Learning (`MLP`) comparisons, per-class F1 heatmaps, and confusion matrices.
- **Production-Ready Artifacts:** Exports trained models, scalers, and encoders via `joblib` and `.keras` for rapid deployment and real-time inference.

---

## 📂 Project Structure

```text
An-Ensemble-Learning-Based-NIDS-CICIDS2017/
│
├── data/
│   ├── raw/                              # Place raw CICIDS2017 CSV files here
│   └── processed/                        # Cleaned, scaled, and split dataset artifacts
│
├── models/                               # Exported pre-trained models
│   ├── decision_tree.pkl
│   ├── hybrid_soft_voting_ids.pkl
│   ├── lightweight_mlp_ids.keras
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── xgboost.pkl
│
├── results/                              # Evaluation metrics, logs, and visualizations
│   ├── manuscript_tables/
│   ├── ablation_paper_table.csv
│   ├── ablation_study_results.csv
│   ├── ablation_study_timing.csv
│   ├── decision_tree_confusion_matrix.csv
│   ├── final_ablation_summary.csv
│   ├── final_computational_cost_table.csv
│   ├── final_literature_comparison.csv
│   ├── final_performance_table.csv
│   ├── hybrid_confusion_matrix.csv
│   ├── hybrid_model_results.csv
│   └── hybrid_model_timing.csv
│
├── src/                                  # Source code scripts for the pipeline
│   ├── data_loader.py                    # Loads raw CSVs and validates schema
│   ├── preprocessing.py                  # Merges data, cleans nulls/infinities/duplicates
│   ├── feature_selection.py              # SelectKBest (ANOVA F-test) top 30 feature extraction
│   ├── scaling.py                        # StandardScaler (zero mean, unit variance)
│   ├── apply_smote.py                    # SMOTE oversampling on training split
│   ├── smote_analysis.py                 # Analyzes class distribution pre/post SMOTE
│   ├── ablation_study.py                 # Evaluates individual and combined model variants
│   ├── plot_ablation_results.py          # Generates visual charts for ablation studies
│   ├── prepare_dl_data.py                # Formats data for Deep Learning (MLP) comparison
│   ├── compare_ml_dl.py                  # Benchmarks Hybrid Ensemble against Lightweight MLP
│   ├── literature_comparison.py          # Compares metrics against existing published studies
│   ├── final_performance_analysis.py     # Generates final metrics, heatmaps & confusion matrices
│   └── prepare_manuscript_results.py     # Compiles tables ready for LaTeX/manuscript inclusion
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 📊 Dataset Information (CICIDS2017)

The system is trained and evaluated on the **CICIDS2017** dataset generated by the Canadian Institute for Cybersecurity. Specifically, the following traffic capture files are utilized:
- `Monday-WorkingHours.pcap_ISCX.csv`
- `Tuesday-WorkingHours.pcap_ISCX.csv`
- `Wednesday-WorkingHours.pcap_ISCX.csv`
- `Friday-WorkingHours.pcap_ISCX.csv`

### Data Summary
- **Initial Raw Records:** $1,050,589$ flows across $79$ features
- **Cleaned Valid Samples:** $992,665$ records (after removing NaNs, infinities, and duplicates)
- **SMOTE Balanced Training Samples:** $6,463,254$ synthetic & real samples
- **Traffic Classes (9 Categories):** `BENIGN`, `Bot`, `DoS Hulk`, `DoS Slowhttptest`, `DoS slowloris`, `DoS GoldenEye`, `FTP-Patator`, `SSH-Patator`, `Heartbleed`

> **📥 Download:** You can download the official CICIDS2017 dataset from the [University of New Brunswick (UNB) CIC Dataset Portal](https://www.unb.ca/cic/datasets/ids-2017.html) or Kaggle, and place the `.csv` files inside `data/raw/`.

---

## ⚙️ Methodology & Hyperparameters

### Pipeline Workflow
1. **Data Cleaning:** Removal of duplicate rows, coercion of non-numeric entries, and filtering of `NaN` and `Inf` values.
2. **Feature Engineering:** Label encoding of target classes and `SelectKBest` ANOVA F-test selecting the top $30$ features.
3. **Stratified Split:** $80\%$ Training, $10\%$ Validation (with 5-fold Stratified Cross-Validation), and $10\%$ Testing (`random_state=42`).
4. **Class Balancing:** `SMOTE` applied exclusively to the $80\%$ training set to prevent data leakage.
5. **Feature Scaling:** `StandardScaler` fitted on training data and applied across train, validation, and test sets.
6. **Soft Voting Fusion:** Combines probability vectors from all four base learners:
   $$\hat{y} = \arg\max_c \frac{1}{4} \sum_{m=1}^{4} P_m(y = c \mid x)$$

### Base Classifier Hyperparameters
| Model | Hyperparameter | Value |
| :--- | :--- | :--- |
| **Logistic Regression** | `max_iter` | `3000` |
| **Decision Tree** | `random_state` | `42` |
| **Random Forest** | `n_estimators` / `max_depth` / `random_state` | `300` / `None` / `42` |
| **XGBoost** | `n_estimators` / `max_depth` / `learning_rate` / `subsample` | `300` / `8` / `0.1` / `0.8` |

---

## 🚀 Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/An-Ensemble-Learning-Based-NIDS-CICIDS2017.git
cd An-Ensemble-Learning-Based-NIDS-CICIDS2017
```

### 2. Set Up Virtual Environment
```bash
python -m venv .venv
# Activate on Windows:
.venv\Scripts\activate
# Activate on Linux/macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
Create a `requirements.txt` (or install directly) using:
```bash
pip install numpy pandas scikit-learn xgboost imbalanced-learn matplotlib seaborn joblib tensorflow
```

### 4. Run the Pipeline
Place your CICIDS2017 `.csv` files into `data/raw/` and execute the scripts sequentially from the root directory:

```bash
# Step 1: Data Merging and Cleaning
python src/preprocessing.py

# Step 2: Feature Selection & Scaling
python src/feature_selection.py
python src/scaling.py

# Step 3: Class Balancing (SMOTE)
python src/apply_smote.py

# Step 4: Model Training, Ablation & Evaluation
python src/ablation_study.py
python src/compare_ml_dl.py
python src/final_performance_analysis.py
```

---

## 📈 Experimental Results

### Overall Classifier Performance
Our proposed Soft Voting Hybrid Ensemble outperforms all individual base estimators across every macro/weighted evaluation metric on the unseen test split:

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 91.54 | 92.26 | 91.54 | 91.41 |
| **XGBoost** | 99.04 | 99.05 | 99.04 | 99.04 |
| **Decision Tree** | 99.35 | 99.35 | 99.35 | 99.35 |
| **Random Forest** | 99.36 | 99.36 | 99.36 | 99.36 |
| **🏆 Hybrid Ensemble (Proposed)** | **99.64** | **99.64** | **99.64** | **99.64** |

### Comparison with Existing Literature
| Study | Method | Dataset | Accuracy (%) |
| :--- | :--- | :---: | :---: |
| Aburomman et al. (2022) | SVM-KNN-PSO Ensemble | NSL-KDD | 98.30 |
| Ullah et al. (2023) | Hybrid Deep Learning IDS | IoT Dataset | 98.70 |
| Qazi et al. (2023) | Hybrid DL (CNN-LSTM) | CICIDS2017 | 99.20 |
| **Proposed System** | **Soft Voting Ensemble (LR+DT+RF+XGB)** | **CICIDS2017** | **99.64** |

---

## 🔮 Future Work

- **Full-Scale Evaluation:** Expanding validation to the complete ~3M record CICIDS2017 and CIC-IDS2018 datasets.
- **Explainable AI (XAI):** Integrating **SHAP** and **LIME** for transparent, human-interpretable security analyst alerts.
- **Edge & IoT Deployment:** Benchmarking real-time packet inference latency and memory footprint on resource-constrained IoT gateways.
- **Zero-Day Representation Learning:** Exploring Autoencoders and Transfer Learning to boost detection of unseen attack signatures.


