"""Train several regression models, compare them, and save the best one.

Run from the project root:  python -m src.train
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    DEPLOY_MODEL,
    METRICS_PATH,
    MODEL_PATH,
    NUMERIC_FEATURES,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    RANGE_COVERAGE,
    TARGET,
    TEST_SIZE,
)

CV_FOLDS = 5  # each model is trained and tested 5 times on different slices
MIN_CATEGORY_COUNT = 5  # categories seen fewer times are grouped as "infrequent"


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Return the input columns (X) and the price to predict (y)."""
    df = pd.read_csv(PROCESSED_DATA_PATH)
    return df[CATEGORICAL_FEATURES + NUMERIC_FEATURES], df[TARGET]


def build_preprocessor() -> ColumnTransformer:
    """Turn text columns into numbers and put numeric columns on one scale."""
    categorical = OneHotEncoder(
        handle_unknown="infrequent_if_exist",  # an unseen model at prediction time won't crash
        min_frequency=MIN_CATEGORY_COUNT,
    )
    return ColumnTransformer(
        [
            ("categorical", categorical, CATEGORICAL_FEATURES),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ]
    )


def candidate_models() -> dict:
    """The regression algorithms to compare, from simplest to most powerful."""
    models = {
        "ridge_regression": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "gradient_boosting": HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.05, random_state=RANDOM_STATE
        ),
    }
    try:  # xgboost is optional: skip it if it is not installed yet
        from xgboost import XGBRegressor

        models["xgboost"] = XGBRegressor(
            n_estimators=600, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
        )
    except ImportError:
        print("xgboost not installed - skipping it.")
    return models


def build_pipeline(model) -> TransformedTargetRegressor:
    """Preprocessing + model in one object; it learns log(price) but answers in rupees."""
    steps = [("preprocess", build_preprocessor()), ("model", model)]
    # HistGradientBoosting needs a regular (dense) table instead of a compressed one
    if isinstance(model, HistGradientBoostingRegressor):
        steps[0][1].set_params(sparse_threshold=0)
    return TransformedTargetRegressor(
        regressor=Pipeline(steps), func=np.log1p, inverse_func=np.expm1
    )


def compare_models(models: dict, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """Cross-validate every candidate on the training set and rank them."""
    rows = []
    for name, model in models.items():
        scores = cross_validate(
            build_pipeline(model), X_train, y_train, cv=CV_FOLDS,
            scoring={
                "mae": "neg_mean_absolute_error",
                "mape": "neg_mean_absolute_percentage_error",
                "r2": "r2",
            },
        )
        rows.append({
            "model": name,
            "cv_mape_%": -scores["test_mape"].mean() * 100,
            "cv_mae_pkr": -scores["test_mae"].mean(),
            "cv_r2": scores["test_r2"].mean(),
        })
        print(f"  {name:<18} done")
    return pd.DataFrame(rows).sort_values("cv_mape_%").reset_index(drop=True)


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Score the final model on data it has never seen."""
    predictions = model.predict(X_test)
    return {
        "test_mape_%": round(mean_absolute_percentage_error(y_test, predictions) * 100, 2),
        "test_mae_pkr": round(mean_absolute_error(y_test, predictions)),
        "test_r2": round(r2_score(y_test, predictions), 3),
    }


def price_interval(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """How far real prices fall from predictions, as multipliers.

    Example: low_ratio 0.85 and high_ratio 1.17 mean that 80% of unseen cars sold
    for between 85% and 117% of the predicted price. Multiplying a new prediction
    by these gives a range that is right about 8 times in 10.
    """
    ratios = y_test.to_numpy() / model.predict(X_test)
    tail = (1 - RANGE_COVERAGE) / 2
    return {
        "coverage_%": round(RANGE_COVERAGE * 100),
        "low_ratio": round(float(np.quantile(ratios, tail)), 4),
        "high_ratio": round(float(np.quantile(ratios, 1 - tail)), 4),
    }


def main() -> None:
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print(f"Comparing models with {CV_FOLDS}-fold cross-validation...")
    models = candidate_models()
    leaderboard = compare_models(models, X_train, y_train)
    print("\n" + leaderboard.round(3).to_string(index=False))

    cv_winner = leaderboard.loc[0, "model"]
    best_name = DEPLOY_MODEL  # see config.py for why this can differ from cv_winner
    best_model = build_pipeline(models[best_name]).fit(X_train, y_train)
    test_scores = evaluate(best_model, X_test, y_test)
    interval = price_interval(best_model, X_test, y_test)
    print(f"\nBest in cross-validation: {cv_winner}. Deployed: {best_name}")
    print(f"Test set (unseen cars): {test_scores}")
    print(f"Likely range: {interval}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH, compress=3)  # compress: smaller file to deploy

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps({
        "best_model": best_name,
        "cv_winner": cv_winner,
        "test": test_scores,
        "price_interval": interval,
        "leaderboard": leaderboard.round(4).to_dict(orient="records"),
    }, indent=2))
    print(f"\nSaved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
