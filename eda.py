"""
eda.py
Generates exploratory charts:
- Yearly average temperature trend
- Monthly rainfall pattern
- Seasonal temperature boxplot
- Correlation heatmap
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

OUTPUT_DIR = "outputs/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_yearly_temp_trend(df):
    """Line chart: average temperature per year"""
    yearly = df.groupby("year")["temp_mean"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(yearly["year"], yearly["temp_mean"], color="#e74c3c", linewidth=2, marker="o", markersize=4)
    ax.fill_between(yearly["year"], yearly["temp_mean"], alpha=0.1, color="#e74c3c")
    ax.set_title("Yearly Average Temperature Trend (1984–2023)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean Temperature (°C)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/01_yearly_temp_trend.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[✓] Saved: {path}")


def plot_monthly_rainfall(df):
    """Bar chart: average rainfall by month"""
    monthly = df.groupby("month")["rainfall_mm"].mean().reset_index()
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly["month_name"] = month_names

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(monthly["month_name"], monthly["rainfall_mm"],
                  color=sns.color_palette("Blues_d", 12))
    ax.set_title("Average Monthly Rainfall Pattern", fontsize=15, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Rainfall (mm)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/02_monthly_rainfall.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[✓] Saved: {path}")


def plot_seasonal_temp_boxplot(df):
    """Boxplot: temperature distribution by season"""
    season_order = ["Winter", "Pre-Monsoon", "Monsoon", "Post-Monsoon"]
    palette = {"Winter": "#3498db", "Pre-Monsoon": "#e67e22",
               "Monsoon": "#2ecc71", "Post-Monsoon": "#9b59b6"}

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="season", y="temp_mean", order=season_order,
                palette=palette, ax=ax)
    ax.set_title("Temperature Distribution by Season", fontsize=15, fontweight="bold")
    ax.set_xlabel("Season")
    ax.set_ylabel("Mean Temperature (°C)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/03_seasonal_temp_boxplot.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[✓] Saved: {path}")


def plot_correlation_heatmap(df):
    """Heatmap: correlations between numeric variables"""
    cols = ["temp_max", "temp_min", "temp_mean", "rainfall_mm", "humidity_pct"]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                square=True, linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap — Climate Variables", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/04_correlation_heatmap.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[✓] Saved: {path}")


def plot_decade_comparison(df):
    """Bar chart: decade-wise average temperature"""
    decade_avg = df.groupby("decade_label")["temp_mean"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(decade_avg["decade_label"], decade_avg["temp_mean"],
           color=["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"])
    ax.set_title("Decade-wise Average Temperature", fontsize=14, fontweight="bold")
    ax.set_xlabel("Decade")
    ax.set_ylabel("Mean Temperature (°C)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/05_decade_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[✓] Saved: {path}")


def run_eda(df):
    print("[INFO] Running EDA charts...")
    plot_yearly_temp_trend(df)
    plot_monthly_rainfall(df)
    plot_seasonal_temp_boxplot(df)
    plot_correlation_heatmap(df)
    plot_decade_comparison(df)
    print("[✓] All EDA charts generated.")


if __name__ == "__main__":
    df = pd.read_csv("data/processed/climate_clean.csv", parse_dates=["date"])
    df["decade"] = (df["year"] // 10) * 10
    df["decade_label"] = df["decade"].astype(str) + "s"
    run_eda(df)