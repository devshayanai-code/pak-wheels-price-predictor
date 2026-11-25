"""Export a browser-ready model and data for the static website in docs/.

GitHub Pages can only serve files, not run Python. So we take the model that
src/train.py saved (the same one the API serves) and write everything the
browser needs to repeat its maths in JavaScript:

    docs/model.json     encoder columns, scaler numbers and every tree
    docs/options.json   dropdown values (makes -> models, fuels, cities...)
    docs/listings.json  compact real listings, for "similar cars"

Run from the project root, after training:  python -m src.export_web
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

from src.config import (
    CATEGORICAL_FEATURES,
    METRICS_PATH,
    MODEL_PATH,
    NUMERIC_FEATURES,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    ROOT_DIR,
    TARGET,
    TEST_SIZE,
)

DOCS_DIR = ROOT_DIR / "docs"


def export_trees(booster: HistGradientBoostingRegressor) -> list[list[list]]:
    """Each tree node -> [feature, threshold, left, right, missing_left, leaf_value].

    A leaf has feature = -1. Thresholds keep full precision: many sit exactly on
    a real value (e.g. mileage 70,000), so rounding them would flip decisions.
    """
    trees = []
    for (predictor,) in booster._predictors:  # one tree per boosting round
        nodes = []
        for node in predictor.nodes:
            if node["is_leaf"]:
                nodes.append([-1, 0, 0, 0, 0, float(node["value"])])
            else:
                nodes.append([
                    int(node["feature_idx"]), float(node["num_threshold"]),
                    int(node["left"]), int(node["right"]), int(node["missing_go_to_left"]), 0,
                ])
        trees.append(nodes)
    return trees


def export_model(pipeline, metrics: dict) -> dict:
    preprocess = pipeline.regressor_.named_steps["preprocess"]
    booster = pipeline.regressor_.named_steps["model"]
    scaler = preprocess.named_transformers_["numeric"]
    return {
        "columns": preprocess.get_feature_names_out().tolist(),  # e.g. "categorical__make_Honda"
        "categorical": CATEGORICAL_FEATURES,
        "numeric": NUMERIC_FEATURES,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "baseline": float(np.ravel(booster._baseline_prediction)[0]),
        "trees": export_trees(booster),
        "target_transform": "log1p",  # the model predicts log(1 + price)
        "metrics": metrics["test"],
        "price_interval": metrics["price_interval"],
    }


def export_options(df: pd.DataFrame) -> dict:
    """Same dropdown values as services.get_options() in the Django API."""
    return {
        "makes": {make: sorted(g["model"].unique().tolist()) for make, g in df.groupby("make")},
        "fuels": sorted(df["fuel"].unique().tolist()),
        "transmissions": sorted(df["transmission"].unique().tolist()),
        "cities": sorted(df["city"].unique().tolist()),
        "year_range": [int(df["year"].min()), int(df["year"].max())],
        "n_listings": len(df),
        "n_cities": int(df.loc[df["city"] != "Other", "city"].nunique()),
    }


def export_listings(df: pd.DataFrame) -> list[list]:
    """[make, model, year, mileage, city, price] rows - lists are smaller than dicts."""
    cols = ["make", "model", "year", "mileage", "city", TARGET]
    return df[cols].astype({"year": int, "mileage": int, TARGET: int}).values.tolist()


def write_json(name: str, data) -> None:
    path = DOCS_DIR / name
    path.write_text(json.dumps(data, separators=(",", ":")))
    print(f"  {path.relative_to(ROOT_DIR)}  {path.stat().st_size / 1024:,.0f} KB")


def main() -> None:
    pipeline = joblib.load(MODEL_PATH)
    booster = pipeline.regressor_.named_steps["model"]
    if not isinstance(booster, HistGradientBoostingRegressor):
        raise SystemExit("The saved model is not gradient boosting: set DEPLOY_MODEL in "
                         "src/config.py and run python -m src.train first.")
    metrics = json.loads(METRICS_PATH.read_text())

    df = pd.read_csv(PROCESSED_DATA_PATH)
    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    X_train, X_test = train_test_split(X, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    metrics["test"]["n_train"], metrics["test"]["n_test"] = len(X_train), len(X_test)

    DOCS_DIR.mkdir(exist_ok=True)
    write_json("model.json", export_model(pipeline, metrics))
    write_json("options.json", export_options(df))
    write_json("listings.json", export_listings(df))

    # Reference predictions so tests can check the JavaScript gives the same answers
    sample = X_test.head(50)
    write_json("reference.json", {
        "cars": sample.to_dict(orient="records"),
        "prices": pipeline.predict(sample).tolist(),
    })


if __name__ == "__main__":
    main()
