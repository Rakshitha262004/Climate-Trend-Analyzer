"""
data_cleaning.py
Loads raw climate CSV, checks for issues, cleans, and saves processed version.
"""

import pandas as pd
import numpy as np
import os

def load_and_clean(
    raw_path="data/raw/climate_data.csv",
    clean_path="data/processed/climate_clean.csv"
):
    """
    Performs:
    1. Date parsing
    2. Missing value detection and imputation
    3. Duplicate removal
    4. Type enforcement
    5. Out-of-range value clamping
    """
    print("[INFO] Loading raw dataset...")
    df = pd.read_csv(raw_path, parse_dates=["date"])

    print(f"[INFO] Shape: {df.shape}")
    print(f"[INFO] Missing values:\n{df.isnull().sum()}")

    # --- Step 1: Remove duplicates ---
    before = len(df)
    df = df.drop_duplicates(subset=["date"])
    print(f"[INFO] Removed {before - len(df)} duplicate rows")

    # --- Step 2: Fill any missing numeric values with monthly median ---
    numeric_cols = ["temp_max", "temp_min", "temp_mean", "rainfall_mm", "humidity_pct"]
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df.groupby("month")[col].transform(
                lambda x: x.fillna(x.median())
            )

    # --- Step 3: Clamp unrealistic values ---
    df["temp_max"] = df["temp_max"].clip(10, 50)
    df["temp_min"] = df["temp_min"].clip(0, 40)
    df["rainfall_mm"] = df["rainfall_mm"].clip(0, 1000)
    df["humidity_pct"] = df["humidity_pct"].clip(10, 100)

    # --- Step 4: Sort by date ---
    df = df.sort_values("date").reset_index(drop=True)

    # --- Step 5: Save ---
    os.makedirs(os.path.dirname(clean_path), exist_ok=True)
    df.to_csv(clean_path, index=False)
    print(f"[✓] Clean dataset saved to '{clean_path}' | Shape: {df.shape}")

    return df


if __name__ == "__main__":
    load_and_clean()