# ==========================================
# STEP 1 - ENVIRONMENT TEST
# CICIDS2017 Hybrid IDS Research
# ==========================================

import pandas as pd
import numpy as np
import sklearn
import imblearn
import xgboost
import matplotlib
import seaborn
import joblib
import psutil


print("=" * 50)
print("CICIDS2017 HYBRID IDS")
print("Environment Test")
print("=" * 50)

print("\nPython environment: OK")
print("Pandas version:", pd.__version__)
print("NumPy version:", np.__version__)
print("Scikit-learn version:", sklearn.__version__)
print("Imbalanced-learn version:", imblearn.__version__)
print("XGBoost version:", xgboost.__version__)
print("Matplotlib version:", matplotlib.__version__)
print("Seaborn version:", seaborn.__version__)
print("Joblib version:", joblib.__version__)
print("PSUtil version:", psutil.__version__)

print("\nAll required libraries imported successfully!")

print("\n" + "=" * 50)
print("Environment setup completed.")
print("=" * 50)
