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
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_predict, cross_validate
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


def car_groups(X: pd.DataFrame) -> pd.Series:
    """Label each listing with its car, so reposts of one car share a label.

    The same car is often listed again at a different price. If one copy lands in
    training and another in testing, the model is tested on a car it already saw.
    Rows with identical inputs (make, model, year, mileage, ...) count as one car.
    """
    return X.astype(str).agg("|".join, axis=1)


def split_data(X: pd.DataFrame, y: pd.Series):
    """Hold back 20% of cars for testing, keeping every repost on the same side."""
    groups = car_groups(X)
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    return (X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx],
            groups.iloc[train_idx])


def group_folds() -> GroupKFold:
    return GroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


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
        # min_samples_leaf=5 (default 20) lets trees learn car models with few
        # listings; chosen with grouped cross-validation on the training set only
        "gradient_boosting": HistGradientBoostingRegressor(
            max_iter=700, learning_rate=0.05, min_samples_leaf=5, random_state=RANDOM_STATE
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


def compare_models(models: dict, X_train: pd.DataFrame, y_train: pd.Series,
                   groups: pd.Series) -> pd.DataFrame:
    """Cross-validate every candidate on the training set and rank them."""
    rows = []
    for name, model in models.items():
        scores = cross_validate(
            build_pipeline(model), X_train, y_train, groups=groups, cv=group_folds(),
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


def price_interval(model, X_train: pd.DataFrame, y_train: pd.Series, groups: pd.Series,
                   X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """How far real prices fall from predictions, as multipliers.

    Example: low_ratio 0.85 and high_ratio 1.17 mean that 80% of cars sold for
    between 85% and 117% of the predicted price. The multipliers are measured on
    the training cars, each predicted by a model that never saw it (cross-validation),
    and then checked on the untouched test cars ("test_coverage_%").
    """
    oof = cross_val_predict(model, X_train, y_train, groups=groups, cv=group_folds())
    ratios = y_train.to_numpy() / oof
    tail = (1 - RANGE_COVERAGE) / 2
    low, high = np.quantile(ratios, tail), np.quantile(ratios, 1 - tail)
    test_ratios = y_test.to_numpy() / model.fit(X_train, y_train).predict(X_test)
    return {
        "coverage_%": round(RANGE_COVERAGE * 100),
        "low_ratio": round(float(low), 4),
        "high_ratio": round(float(high), 4),
        "test_coverage_%": round(float(np.mean((test_ratios >= low) & (test_ratios <= high))) * 100, 1),
    }


def main() -> None:
    X, y = load_data()
    X_train, X_test, y_train, y_test, groups = split_data(X, y)
    print(f"{len(X_train):,} training and {len(X_test):,} test listings "
          "(reposts of one car never appear on both sides)")

    print(f"Comparing models with {CV_FOLDS}-fold grouped cross-validation...")
    models = candidate_models()
    leaderboard = compare_models(models, X_train, y_train, groups)
    print("\n" + leaderboard.round(3).to_string(index=False))

    cv_winner = leaderboard.loc[0, "model"]
    best_name = DEPLOY_MODEL  # see config.py for why this can differ from cv_winner
    best_model = build_pipeline(models[best_name])
    interval = price_interval(best_model, X_train, y_train, groups, X_test, y_test)
    test_scores = evaluate(best_model, X_test, y_test)  # price_interval fitted it
    print(f"\nBest in cross-validation: {cv_winner}. Deployed: {best_name}")
    print(f"Test set (unseen cars): {test_scores}")

    # The previous release's settings, scored on the same test cars for a fair comparison
    previous = build_pipeline(HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.05, random_state=RANDOM_STATE)).fit(X_train, y_train)
    previous_scores = evaluate(previous, X_test, y_test)
    print(f"Previous settings on the same test cars: {previous_scores}")
    print(f"Likely range: {interval}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH, compress=3)  # compress: smaller file to deploy

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps({
        "best_model": best_name,
        "cv_winner": cv_winner,
        "test": test_scores,
        "previous_settings_test": previous_scores,
        "price_interval": interval,
        "leaderboard": leaderboard.round(4).to_dict(orient="records"),
    }, indent=2))
    print(f"\nSaved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
