"""Clean the raw PakWheels.com used-car listings into a model-ready dataset.

Run from the project root:  python -m src.data_cleaning
"""
import pandas as pd

from src.config import PROCESSED_DATA_PATH, RAW_DATA_PATH, TARGET

# Cities with fewer listings than this are grouped as "Other"
MIN_CITY_LISTINGS = 50

# Outlier rules: rows outside these limits are typos or rare special cases
MIN_PRICE = 100_000  # below this the price is a typo (e.g. PKR 1,100)
MIN_YEAR = 1980  # vintage cars are priced as collectibles, not by usual rules
MAX_MILEAGE = 500_000  # values like 999,999 km are placeholders, not real readings
MAX_ENGINE_CC = 7_000  # larger values (e.g. 15,000 cc) are typing mistakes

# Price typos: compared with the median of the same make, model and year (only
# when at least MIN_GROUP_FOR_TYPOS such cars exist). A missing zero gives about
# 0.1x the usual price, so anything below 0.2x or above 5x is treated as a typo.
MIN_GROUP_FOR_TYPOS = 5
TYPO_LOW_RATIO, TYPO_HIGH_RATIO = 0.2, 5.0

# Fake mileage: sellers type "1" or "10" km on cars that are years old
MIN_REAL_MILEAGE = 100  # km
FAKE_MILEAGE_MIN_AGE = 3  # years older than the newest car in the data

# Car makes whose name is two words in the URL (e.g. "mercedes-benz-c-class")
MULTI_WORD_MAKES = ("mercedes-benz", "land-rover", "range-rover")

# Pulls "honda-civic" out of ".../used-cars/honda-civic-2013-for-sale-in-..."
URL_SLUG_PATTERN = r"used-cars/(?P<slug>.+)-\d{4}-for-sale"


def load_raw_data(path=RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV exactly as downloaded."""
    return pd.read_csv(path)


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per listing; the same car appears under several categories."""
    return df.drop_duplicates(subset="url").reset_index(drop=True)


def clean_price(df: pd.DataFrame) -> pd.DataFrame:
    """Turn 'PKR 3,650,000' into 3650000 and drop rows with no usable price."""
    df = df.copy()
    digits_only = df[TARGET].astype("string").str.replace(r"[^\d]", "", regex=True)
    df[TARGET] = pd.to_numeric(digits_only, errors="coerce")  # "Call for price" -> NaN
    return df.dropna(subset=[TARGET]).reset_index(drop=True)


def clean_mileage(df: pd.DataFrame) -> pd.DataFrame:
    """Turn '138,000 km' into 138000."""
    df = df.copy()
    digits_only = df["mileage"].str.replace(r"[^\d]", "", regex=True)
    df["mileage"] = pd.to_numeric(digits_only, errors="coerce")
    return df


def clean_engine(df: pd.DataFrame) -> pd.DataFrame:
    """Turn '1800 cc' into engine_cc = 1800. Electric cars get 0.

    Battery sizes (kWh) are not kept: sellers often typed the driving range
    instead (e.g. 400 for a Nissan Leaf), so the values cannot be trusted.
    The 'fuel' column already tells the model which cars are electric.
    """
    df = df.copy()
    cc = df["engine"].str.extract(r"([\d.]+)\s*cc", expand=False)
    df["engine_cc"] = pd.to_numeric(cc).fillna(0)
    return df.drop(columns="engine")


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with impossible or untypical values (limits defined at the top)."""
    is_electric = df["fuel"] == "Electric"
    keep = (
        (df[TARGET] >= MIN_PRICE)
        & (df["year"] >= MIN_YEAR)
        & (df["mileage"] <= MAX_MILEAGE)
        & (df["engine_cc"] <= MAX_ENGINE_CC)
        & (is_electric | (df["engine_cc"] > 0))  # fuel cars must have an engine size
    )
    return df[keep].reset_index(drop=True)


def _split_slug(slug: str) -> tuple[str, str]:
    """Split a URL slug like 'mercedes-benz-c-class' into ('Mercedes-Benz', 'C Class')."""
    for make in MULTI_WORD_MAKES:
        if slug.startswith(make + "-"):
            model = slug[len(make) + 1 :]
            return make.title(), model.replace("-", " ").title()
    make, _, model = slug.partition("-")
    return make.title(), model.replace("-", " ").title()


def extract_make_model(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'make' and 'model' columns, taken from the listing URL."""
    df = df.copy()
    slugs = df["url"].str.extract(URL_SLUG_PATTERN)["slug"]
    df[["make", "model"]] = slugs.apply(_split_slug).tolist()
    return df


def group_rare_cities(df: pd.DataFrame, min_count: int = MIN_CITY_LISTINGS) -> pd.DataFrame:
    """Replace cities with fewer than `min_count` listings by 'Other'."""
    df = df.copy()
    counts = df["city"].value_counts()
    rare = counts[counts < min_count].index
    df.loc[df["city"].isin(rare), "city"] = "Other"
    return df


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the model inputs and the target, in a readable order."""
    columns = [
        "make", "model", "year", "mileage", "engine_cc",
        "fuel", "transmission", "city", TARGET,
    ]
    return df[columns]


def drop_reposts(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that are identical in every column (the same car posted twice)."""
    return df.drop_duplicates().reset_index(drop=True)


def remove_price_typos(df: pd.DataFrame) -> pd.DataFrame:
    """Drop prices far from what the same make, model and year usually sell for."""
    group = df.groupby(["make", "model", "year"])[TARGET]
    ratio = df[TARGET] / group.transform("median")
    checkable = group.transform("size") >= MIN_GROUP_FOR_TYPOS
    typo = checkable & ((ratio < TYPO_LOW_RATIO) | (ratio > TYPO_HIGH_RATIO))
    return df[~typo].reset_index(drop=True)


def remove_fake_mileage(df: pd.DataFrame) -> pd.DataFrame:
    """Drop cars several years old that claim almost no kilometres (e.g. "1 km")."""
    old = df["year"] <= df["year"].max() - FAKE_MILEAGE_MIN_AGE
    fake = old & (df["mileage"] < MIN_REAL_MILEAGE)
    return df[~fake].reset_index(drop=True)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run every cleaning step in order (the order matters: duplicates go first)."""
    return (
        df.pipe(remove_duplicates)
        .pipe(clean_price)
        .pipe(clean_mileage)
        .pipe(clean_engine)
        .pipe(remove_outliers)
        .pipe(extract_make_model)
        .pipe(group_rare_cities)
        .pipe(select_columns)
        .pipe(drop_reposts)
        .pipe(remove_price_typos)
        .pipe(remove_fake_mileage)
    )


def main() -> None:
    raw = load_raw_data()
    clean = clean_data(raw)

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"Raw rows:   {len(raw):,}")
    print(f"Clean rows: {len(clean):,}")
    print(f"Saved to:   {PROCESSED_DATA_PATH}\n")
    print(clean.head().to_string())


if __name__ == "__main__":
    main()
