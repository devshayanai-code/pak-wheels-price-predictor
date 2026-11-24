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

# Training settings
TARGET = "price"  # column we predict; confirmed after inspecting the data
TEST_SIZE = 0.2  # 20% of rows held back to test the model
RANDOM_STATE = 42  # fixed seed so results are repeatable
