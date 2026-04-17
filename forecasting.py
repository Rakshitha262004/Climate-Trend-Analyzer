"""
forecasting.py
Forecasts annual mean temperature for the next 5 years using:
1. Linear regression extrapolation (primary)
2. Basic ARIMA (optional, if statsmodels is available)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os
import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR = "outputs/images"
TABLE_DIR = "outputs/tables"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)


def linear_forecast(df, forecast_years=5):
    """
    Extrapolates the linear temperature trend into the future.
    """
    yearly = df.groupby("year")["temp_mean"].mean().reset_index()
    X = yearly["year"].values.reshape(-1, 1)
    y = yearly["temp_mean"].values

    model = LinearRegression()
    model.fit(X, y)

    last_year = int(yearly["year"].max())
    future_years = np.arange(last_year + 1, last_year + forecast_years + 1).reshape(-1, 1)
    future_preds = model.predict(future_years)
    historical_preds = model.predict(X)

    # Build forecast dataframe
    forecast_df = pd.DataFrame({
        "year": future_years.flatten(),
        "forecasted_temp": np.round(future_preds, 3),
        "method": "LinearRegression"
    })
    forecast_df.to_csv(f"{TABLE_DIR}/temperature_forecast.csv", index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(yearly["year"], yearly["temp_mean"], color="#e74c3c",
            linewidth=2, label="Historical Annual Mean Temp")
    ax.plot(yearly["year"], historical_preds, color="#2c3e50",
            linewidth=1.5, linestyle="--", label="Historical Trend Fit")
    ax.plot(future_years.flatten(), future_preds, color="#f39c12",
            linewidth=2.5, linestyle="--", marker="o", markersize=6,
            label=f"Forecast ({last_year+1}–{last_year+forecast_years})")

    # Confidence band (simple ±1 std of residuals)
    residuals = y - historical_preds
    std_res = np.std(residuals)
    ax.fill_between(future_years.flatten(),
                    future_preds - std_res, future_preds + std_res,
                    alpha=0.2, color="#f39c12", label="±1σ Confidence Band")

    ax.axvline(x=last_year + 0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_title("Temperature Forecast — Next 5 Years (Linear Regression)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean Temperature (°C)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/10_temperature_forecast.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[✓] Forecast chart saved: {path}")
    print(forecast_df.to_string(index=False))
    return forecast_df


def arima_forecast(df, forecast_months=24):
    """
    Optional: ARIMA-based monthly temperature forecast.
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        print("[WARN] statsmodels not available for ARIMA. Skipping.")
        return

    ts = df.set_index("date")["temp_mean"].asfreq("MS").interpolate()

    model = ARIMA(ts, order=(2, 1, 2))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_months)

    fig, ax = plt.subplots(figsize=(14, 5))
    ts[-60:].plot(ax=ax, color="#e74c3c", label="Historical (last 5yr)")
    forecast.plot(ax=ax, color="#f39c12", linestyle="--", label="ARIMA Forecast (24 months)")
    ax.set_title("ARIMA Temperature Forecast — Next 24 Months", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Mean Temperature (°C)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/11_arima_forecast.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[✓] ARIMA forecast chart saved: {path}")


def run_forecasting(df):
    print("[INFO] Running forecasting...")
    linear_forecast(df)
    arima_forecast(df)
    print("[✓] Forecasting complete.")


if __name__ == "__main__":
    df = pd.read_csv("data/processed/climate_clean.csv", parse_dates=["date"])
    run_forecasting(df)