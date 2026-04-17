"""
feature_engineering.py
Adds rolling averages, temperature anomaly scores, and seasonal indicators.
"""

import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
    - 12-month rolling average for temperature and rainfall
    - Temperature anomaly (deviation from long-term mean)
    - Rainfall anomaly
    - Decade label
    """
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    # --- Rolling averages (12-month window) ---
    df["temp_rolling_12m"] = df["temp_mean"].rolling(window=12, min_periods=1).mean()
    df["rain_rolling_12m"] = df["rainfall_mm"].rolling(window=12, min_periods=1).mean()

    # --- Anomaly: deviation from overall monthly mean ---
    monthly_mean_temp = df.groupby("month")["temp_mean"].transform("mean")
    monthly_mean_rain = df.groupby("month")["rainfall_mm"].transform("mean")

    df["temp_anomaly"] = df["temp_mean"] - monthly_mean_temp
    df["rain_anomaly"] = df["rainfall_mm"] - monthly_mean_rain

    # --- Decade label ---
    df["decade"] = (df["year"] // 10) * 10
    df["decade_label"] = df["decade"].astype(str) + "s"

    print("[✓] Feature engineering complete. New columns added.")
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/processed/climate_clean.csv", parse_dates=["date"])
    df = engineer_features(df)
    print(df[["date", "temp_anomaly", "rain_anomaly", "decade_label"]].head(10))