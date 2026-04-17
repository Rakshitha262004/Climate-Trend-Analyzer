"""
main.py
Runs the complete Climate Trend Analyzer pipeline:
1. Generate synthetic dataset
2. Clean data
3. Feature engineering
4. EDA
5. Trend analysis
6. Anomaly detection
7. Forecasting
"""

import os
import pandas as pd

# Create all directories upfront
for d in ["data/raw", "data/processed", "outputs/images", "outputs/tables"]:
    os.makedirs(d, exist_ok=True)

# --- Step 1: Generate Dataset ---
print("\n" + "="*60)
print("STEP 1: GENERATING SYNTHETIC CLIMATE DATASET")
print("="*60)
from src.data_generator import generate_climate_data
df_raw = generate_climate_data()

# --- Step 2: Clean Data ---
print("\n" + "="*60)
print("STEP 2: DATA CLEANING AND PREPROCESSING")
print("="*60)
from src.data_cleaning import load_and_clean
df_clean = load_and_clean()

# --- Step 3: Feature Engineering ---
print("\n" + "="*60)
print("STEP 3: FEATURE ENGINEERING")
print("="*60)
from src.feature_engineering import engineer_features
df = engineer_features(df_clean)

# --- Step 4: EDA ---
print("\n" + "="*60)
print("STEP 4: EXPLORATORY DATA ANALYSIS")
print("="*60)
from src.eda import run_eda
run_eda(df)

# --- Step 5: Trend Analysis ---
print("\n" + "="*60)
print("STEP 5: TREND ANALYSIS")
print("="*60)
from src.trend_analysis import run_trend_analysis
run_trend_analysis(df)

# --- Step 6: Anomaly Detection ---
print("\n" + "="*60)
print("STEP 6: ANOMALY DETECTION")
print("="*60)
from src.anomaly_detection import run_anomaly_detection
run_anomaly_detection(df)

# --- Step 7: Forecasting ---
print("\n" + "="*60)
print("STEP 7: FORECASTING")
print("="*60)
from src.forecasting import run_forecasting
run_forecasting(df)

print("\n" + "="*60)
print("✅ PIPELINE COMPLETE")
print(f"   Charts saved to:  outputs/images/")
print(f"   Tables saved to:  outputs/tables/")
print("="*60)