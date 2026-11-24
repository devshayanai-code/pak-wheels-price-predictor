"""Exploratory data analysis: summary numbers and charts of the cleaned data.

Run from the project root:  python -m src.eda
Charts are saved to reports/figures/.
"""
import matplotlib

matplotlib.use("Agg")  # save charts to files without opening windows

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR, PROCESSED_DATA_PATH, TARGET

TOP_N_MAKES = 10
LAKH = 100_000  # prices are shown in lakh rupees, as on PakWheels


def save_figure(fig: plt.Figure, name: str) -> None:
    """Save a chart as PNG and free its memory."""
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=120)
    plt.close(fig)


def plot_price_distribution(df: pd.DataFrame) -> None:
    """Show why we model log(price): the raw price has a long right tail."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(df[TARGET] / LAKH, bins=60, ax=left)
    left.set(title="Price (raw): most cars cheap, a few very expensive",
             xlabel="Price (lakh PKR)")
    sns.histplot(np.log10(df[TARGET]), bins=60, ax=right)
    right.set(title="log10(price): close to a bell shape", xlabel="log10(price in PKR)")
    save_figure(fig, "01_price_distribution")


def plot_price_vs_year(df: pd.DataFrame) -> None:
    """Newer cars should cost more: check that the data agrees."""
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.scatterplot(data=df, x="year", y=TARGET, alpha=0.2, s=10, ax=ax)
    median_by_year = df.groupby("year")[TARGET].median()
    ax.plot(median_by_year.index, median_by_year.values, color="crimson", label="Median")
    ax.set(yscale="log", title="Price vs year", ylabel="Price (PKR, log scale)")
    ax.legend()
    save_figure(fig, "02_price_vs_year")


def plot_price_vs_mileage(df: pd.DataFrame) -> None:
    """More kilometres driven should mean a lower price."""
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.scatterplot(data=df, x="mileage", y=TARGET, alpha=0.2, s=10, ax=ax)
    ax.set(yscale="log", title="Price vs mileage", xlabel="Mileage (km)",
           ylabel="Price (PKR, log scale)")
    save_figure(fig, "03_price_vs_mileage")


def plot_top_makes(df: pd.DataFrame) -> None:
    """Median price of the most listed makes."""
    top = df["make"].value_counts().head(TOP_N_MAKES).index
    medians = (
        df[df["make"].isin(top)].groupby("make")[TARGET].median().sort_values() / LAKH
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    medians.plot.barh(ax=ax)
    ax.set(title=f"Median price of the {TOP_N_MAKES} most listed makes",
           xlabel="Median price (lakh PKR)", ylabel="")
    save_figure(fig, "04_median_price_by_make")


def print_summary(df: pd.DataFrame) -> None:
    """Print the key numbers in plain words."""
    print(f"Cars: {len(df):,}")
    print(f"Median price: PKR {df[TARGET].median():,.0f}")
    print(f"Price range: PKR {df[TARGET].min():,.0f} to {df[TARGET].max():,.0f}")
    print("\nCorrelation with log(price) (+ = goes up together, - = opposite):")
    numeric = df[["year", "mileage", "engine_cc"]].assign(log_price=np.log10(df[TARGET]))
    print(numeric.corr()["log_price"].drop("log_price").round(2).to_string())


def main() -> None:
    df = pd.read_csv(PROCESSED_DATA_PATH)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print_summary(df)
    plot_price_distribution(df)
    plot_price_vs_year(df)
    plot_price_vs_mileage(df)
    plot_top_makes(df)
    print(f"\nCharts saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
