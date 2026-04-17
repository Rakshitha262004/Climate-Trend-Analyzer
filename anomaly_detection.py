"""
anomaly_detection.py
Detects climate anomalies using:
1. Z-score method (|Z| > 2.5)
2. IQR method
Saves anomaly table and plots anomaly timeline.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "outputs/images"
TABLE_DIR = "outputs/tables"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)


def zscore_anomaly_detection(df, column="temp_mean", threshold=2.5):
    """
    Flags records where Z-score of a column exceeds threshold.
    """
    mean = df[column].mean()
    std = df[column].std()
    df = df.copy()
    df[f"zscore_{column}"] = (df[column] - mean) / std
    df[f"anomaly_zscore_{column}"] = df[f"zscore_{column}"].abs() > threshold
    anomalies = df[df[f"anomaly_zscore_{column}"]]
    print(f"[Z-Score] Found {len(anomalies)} anomalies in '{column}' (threshold={threshold})")
    return df, anomalies


def iqr_anomaly_detection(df, column="rainfall_mm"):
    """
    Flags records where value is below Q1 - 1.5*IQR or above Q3 + 1.5*IQR.
    """
    df = df.copy()
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[f"anomaly_iqr_{column}"] = (df[column] < lower) | (df[column] > upper)
    anomalies = df[df[f"anomaly_iqr_{column}"]]
    print(f"[IQR] Found {len(anomalies)} anomalies in '{column}' (lower={lower:.2f}, upper={upper:.2f})")
    return df, anomalies


def plot_anomaly_timeline(df, df_with_anomaly_col, column, anomaly_col, title, filename):
    """
    Plots a time series with anomaly points highlighted in red.
    """
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(df["date"], df[column], color="#7f8c8d", linewidth=0.8, alpha=0.7, label=column)

    anomalies = df_with_anomaly_col[df_with_anomaly_col[anomaly_col] == True]
    ax.scatter(anomalies["date"], anomalies[column],
               color="red", zorder=5, s=40, label="Anomaly", alpha=0.9)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel(column)
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/{filename}"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[✓] Anomaly chart saved: {path}")


def run_anomaly_detection(df):
    print("[INFO] Running anomaly detection...")

    # Temperature anomaly via Z-score
    df_temp, temp_anomalies = zscore_anomaly_detection(df, column="temp_mean", threshold=2.5)
    plot_anomaly_timeline(
        df, df_temp, "temp_mean", "anomaly_zscore_temp_mean",
        "Temperature Anomalies (Z-score method, threshold=2.5σ)",
        "08_temp_anomalies.png"
    )

    # Rainfall anomaly via IQR
    df_rain, rain_anomalies = iqr_anomaly_detection(df, column="rainfall_mm")
    plot_anomaly_timeline(
        df, df_rain, "rainfall_mm", "anomaly_iqr_rainfall_mm",
        "Rainfall Anomalies (IQR method)",
        "09_rainfall_anomalies.png"
    )

    # Save anomaly tables
    temp_anomalies[["date", "year", "month", "temp_mean", "zscore_temp_mean"]].to_csv(
        f"{TABLE_DIR}/temp_anomalies.csv", index=False
    )
    rain_anomalies[["date", "year", "month", "rainfall_mm"]].to_csv(
        f"{TABLE_DIR}/rain_anomalies.csv", index=False
    )

    print(f"[✓] Anomaly detection complete. Tables saved.")
    return df_temp, df_rain


if __name__ == "__main__":
    df = pd.read_csv("data/processed/climate_clean.csv", parse_dates=["date"])
    run_anomaly_detection(df)