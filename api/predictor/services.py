"""The ML logic behind the API: load the model once, predict a price, find similar cars.

Views stay thin and only handle HTTP; everything about the model lives here.
"""
import json
from functools import lru_cache

import joblib
import pandas as pd

from src.config import (
    CATEGORICAL_FEATURES,
    METRICS_PATH,
    MODEL_PATH,
    NUMERIC_FEATURES,
    PROCESSED_DATA_PATH,
    TARGET,
)

MAX_SIMILAR_LISTINGS = 3
SIMILAR_YEAR_WINDOW = 2  # "similar" = same make and model, at most 2 years apart
KM_PER_YEAR_OF_DIFFERENCE = 50_000  # 50,000 km apart counts like 1 year apart
ROUND_TO = 10_000  # round prices to the nearest 10,000 rupees, like real listings


@lru_cache(maxsize=1)
def get_model():
    """Load the trained pipeline once and keep it in memory."""
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def get_listings() -> pd.DataFrame:
    """Load the cleaned listings once (used for similar cars and dropdown options)."""
    return pd.read_csv(PROCESSED_DATA_PATH)


@lru_cache(maxsize=1)
def get_metrics() -> dict:
    """Load the saved test scores of the model."""
    return json.loads(METRICS_PATH.read_text())


def round_price(value: float) -> int:
    return int(round(value / ROUND_TO) * ROUND_TO)


def get_options() -> dict:
    """Valid values for each input, e.g. to fill dropdown menus in the frontend."""
    df = get_listings()
    models_by_make = {
        make: sorted(group["model"].unique().tolist())
        for make, group in df.groupby("make")
    }
    return {
        "makes": models_by_make,
        "fuels": sorted(df["fuel"].unique().tolist()),
        "transmissions": sorted(df["transmission"].unique().tolist()),
        "cities": sorted(df["city"].unique().tolist()),
        "year_range": [int(df["year"].min()), int(df["year"].max())],
    }


def find_similar_listings(car: dict, limit: int = MAX_SIMILAR_LISTINGS) -> list[dict]:
    """Real listings of the same make and model that are closest in year and mileage."""
    df = get_listings()
    same_car = df[
        (df["make"] == car["make"])
        & (df["model"] == car["model"])
        & ((df["year"] - car["year"]).abs() <= SIMILAR_YEAR_WINDOW)
    ]
    distance = (same_car["year"] - car["year"]).abs() + (
        (same_car["mileage"] - car["mileage"]).abs() / KM_PER_YEAR_OF_DIFFERENCE
    )
    closest = same_car.assign(distance=distance).nsmallest(limit, "distance")
    return (
        closest[["year", "mileage", "city", TARGET]]
        .astype({"year": int, "mileage": int, TARGET: int})
        .rename(columns={TARGET: "price"})
        .to_dict(orient="records")
    )


def predict_price(car: dict) -> dict:
    """Predict the price of one car, with a likely range and similar real listings."""
    row = pd.DataFrame([car])[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    price = float(get_model().predict(row)[0])

    # The range uses the model's average error on unseen cars (e.g. +/- 15%)
    error_rate = get_metrics()["test"]["test_mape_%"] / 100
    return {
        "predicted_price": round_price(price),
        "price_range": {
            "low": round_price(price * (1 - error_rate)),
            "high": round_price(price * (1 + error_rate)),
        },
        "average_error_percent": round(error_rate * 100, 1),
        "similar_listings": find_similar_listings(car),
    }
