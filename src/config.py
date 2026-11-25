"""Central place for project paths and settings.

Every other file imports from here, so a path is changed in one place only.
"""
from pathlib import Path

# Project root = the folder that contains src/
ROOT_DIR = Path(__file__).resolve().parent.parent

# Data
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "pakwheels.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "pakwheels_clean.csv"

# Charts
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

# Model
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "car_price_model.joblib"

METRICS_PATH = ROOT_DIR / "reports" / "metrics.json"

# Model inputs
CATEGORICAL_FEATURES = ["make", "model", "fuel", "transmission", "city"]
NUMERIC_FEATURES = ["year", "mileage", "engine_cc"]

# Training settings
TARGET = "price"  # column we predict; confirmed after inspecting the data
TEST_SIZE = 0.2  # 20% of rows held back to test the model
RANDOM_STATE = 42  # fixed seed so results are repeatable

# Model that gets saved and served by both the API and the website. It must be
# gradient boosting: the website runs it in the browser (src/export_web.py), and
# a 200-tree random forest would be far too large to download. Gradient boosting
# also has the lowest error in grouped cross-validation (see reports/metrics.json).
DEPLOY_MODEL = "gradient_boosting"
RANGE_COVERAGE = 0.80  # the "likely range" should contain 80% of real prices
LOW_DATA_LISTINGS = 30  # fewer listings of a car model than this: estimate is less reliable
