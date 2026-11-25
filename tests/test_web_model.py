"""The website runs the model in JavaScript; check it gives Python's exact answers."""
import json
import shutil
import subprocess

import pytest

from src.config import ROOT_DIR

DOCS = ROOT_DIR / "docs"

CHECK_SCRIPT = """
const P = require("./docs/predictor.js");
const model = require("./docs/model.json");
const ref = require("./docs/reference.json");
const worst = Math.max(...ref.cars.map((car, i) =>
  Math.abs(P.predictRaw(model, car) - ref.prices[i]) / ref.prices[i]));
console.log(JSON.stringify({ worst }));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_javascript_matches_python():
    result = subprocess.run(
        ["node", "-e", CHECK_SCRIPT], cwd=ROOT_DIR, capture_output=True, text=True, check=True
    )
    assert json.loads(result.stdout)["worst"] < 1e-9


def test_site_files_exist():
    for name in ["index.html", "predictor.js", "model.json", "options.json", "listings.json"]:
        assert (DOCS / name).exists(), f"docs/{name} is missing: run python -m src.export_web"


def test_website_and_api_use_the_same_model():
    from src.config import DEPLOY_MODEL, METRICS_PATH

    metrics = json.loads(METRICS_PATH.read_text())
    site = json.loads((DOCS / "model.json").read_text())
    assert metrics["best_model"] == DEPLOY_MODEL
    assert site["metrics"]["test_mape_%"] == metrics["test"]["test_mape_%"]
    assert site["price_interval"] == metrics["price_interval"]


def test_likely_range_surrounds_the_estimate():
    from src.config import METRICS_PATH

    interval = json.loads(METRICS_PATH.read_text())["price_interval"]
    assert interval["coverage_%"] == 80
    assert interval["low_ratio"] < 1 < interval["high_ratio"]
    # Measured on training cars, then checked on untouched test cars
    assert 70 <= interval["test_coverage_%"] <= 90


def test_reposts_never_cross_the_train_test_split():
    import pandas as pd

    from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, PROCESSED_DATA_PATH, TARGET
    from src.train import car_groups, split_data

    df = pd.read_csv(PROCESSED_DATA_PATH)
    X_train, X_test, *_ = split_data(df[CATEGORICAL_FEATURES + NUMERIC_FEATURES], df[TARGET])
    assert not set(car_groups(X_train)) & set(car_groups(X_test))
