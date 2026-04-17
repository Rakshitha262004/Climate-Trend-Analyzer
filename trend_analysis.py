"""
trend_analysis.py
Performs:
1. Linear regression on annual mean temperature
2. Seasonal decomposition using statsmodels STL
3. Mann-Kendall trend test
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import STL
import os

try:
    import pymannkendall as mk
    MK_AVAILABLE = True
except ImportError:
    MK_AVAILABLE = False

OUTPUT_DIR = "outputs/images"
TABLE_DIR = "outputs/tables"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)


def linear_trend_analysis(df):
    """
    Fits a linear regression to annual mean temperatures.
    Returns slope (°C per year) and plots the trend.
    """
    yearly = df.groupby("year")["temp_mean"].mean().reset_index()
    X = yearly["year"].values.reshape(-1, 1)
    y = yearly["temp_mean"].values

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    slope = model.coef_[0]
    intercept = model.intercept_
    r2 = model.score(X, y)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.scatter(yearly["year"], yearly["temp_mean"], color="#e74c3c", s=50, label="Annual Mean Temp", zorder=3)
    ax.plot(yearly["year"], y_pred, color="#2c3e50", linewidth=2.5,
            linestyle="--", label=f"Trend Line (slope={slope:.4f}°C/year)")
    ax.set_title("Long-term Temperature Trend — Linear Regression", fontsize=14, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean Temperature (°C)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/06_linear_trend.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[✓] Linear trend chart saved: {path}")

    # Save summary
    summary = pd.DataFrame({
        "metric": ["slope_degC_per_year", "intercept", "r2_score"],
        "value": [round(slope, 5), round(intercept, 4), round(r2, 4)]
    })
    summary.to_csv(f"{TABLE_DIR}/trend_summary.csv", index=False)
    print(f"[✓] Trend summary: slope={slope:.5f}°C/year, R²={r2:.4f}")
    return slope, r2


def seasonal_decomposition(df):
    """
    Uses STL decomposition to separate trend, seasonality, and residuals.
    """
    # Set date as index, use monthly frequency
    ts = df.set_index("date")["temp_mean"].asfreq("MS")

    # Fill any gaps
    ts = ts.interpolate()

    stl = STL(ts, period=12, robust=True)
    result = stl.fit()

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    result.observed.plot(ax=axes[0], color="#2c3e50")
    axes[0].set_title("Observed", fontsize=11)
    axes[0].set_ylabel("°C")

    result.trend.plot(ax=axes[1], color="#e74c3c")
    axes[1].set_title("Trend Component", fontsize=11)
    axes[1].set_ylabel("°C")

    result.seasonal.plot(ax=axes[2], color="#2ecc71")
    axes[2].set_title("Seasonal Component", fontsize=11)
    axes[2].set_ylabel("°C")

    result.resid.plot(ax=axes[3], color="#7f8c8d")
    axes[3].set_title("Residual Component", fontsize=11)
    axes[3].set_ylabel("°C")

    plt.suptitle("STL Seasonal Decomposition — Monthly Temperature", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/07_stl_decomposition.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] STL decomposition chart saved: {path}")


def mann_kendall_test(df):
    """
    Mann-Kendall statistical test for monotonic trend.
    """
    if not MK_AVAILABLE:
        print("[WARN] pymannkendall not installed. Skipping Mann-Kendall test.")
        return

    yearly = df.groupby("year")["temp_mean"].mean()
    result = mk.original_test(yearly)
    print(f"\n[Mann-Kendall Trend Test]")
    print(f"  Trend     : {result.trend}")
    print(f"  p-value   : {result.p:.5f}")
    print(f"  Tau       : {result.Tau:.4f}")
    print(f"  Slope     : {result.slope:.5f}°C/year")
    print(f"  {'Statistically significant trend detected.' if result.p < 0.05 else 'No significant trend detected.'}")

    # Save
    mk_df = pd.DataFrame([{
        "test": "Mann-Kendall",
        "trend": result.trend,
        "p_value": round(result.p, 5),
        "tau": round(result.Tau, 4),
        "slope": round(result.slope, 5),
        "significant": result.p < 0.05
    }])
    mk_df.to_csv(f"{TABLE_DIR}/mann_kendall_result.csv", index=False)


def run_trend_analysis(df):
    print("[INFO] Running trend analysis...")
    linear_trend_analysis(df)
    seasonal_decomposition(df)
    mann_kendall_test(df)
    print("[✓] Trend analysis complete.")


if __name__ == "__main__":
    df = pd.read_csv("data/processed/climate_clean.csv", parse_dates=["date"])
    run_trend_analysis(df)