from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor


CATEGORICAL_COLS = ["realestate_type"]
NUMERIC_COLS = [
    "municipality_number",
    "lat",
    "lon",
    "built_year",
    "bra",
    "total_area",
    "floor",
    "bedrooms",
    "rooms",
    "area_ratio",
]


@dataclass(frozen=True)
class TrainResult:
    pipeline: Any
    metrics: dict[str, Any]
    feature_schema: dict[str, Any]


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["area_ratio"] = df["bra"] / df["total_area"]
    df.loc[~np.isfinite(df["area_ratio"]), "area_ratio"] = np.nan
    return df


def train_and_evaluate(rows: list[dict[str, Any]]) -> TrainResult:
    df = pd.DataFrame(rows)
    df = _add_derived_features(df)

    y = df["price"].astype(float)
    X = df[CATEGORICAL_COLS + NUMERIC_COLS].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, CATEGORICAL_COLS),
            ("num", numeric_transformer, NUMERIC_COLS),
        ]
    )

    model = HistGradientBoostingRegressor(random_state=42)

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))

    # MAE by realestate_type on test set
    test_df = X_test.copy()
    test_df["y_true"] = y_test.values
    test_df["y_pred"] = y_pred

    by_type = {}
    for rt, g in test_df.groupby("realestate_type", sort=False):
        by_type[str(rt)] = float(mean_absolute_error(g["y_true"], g["y_pred"]))

    # MAE by top municipalities by volume in test set
    top_munis = (
        test_df["municipality_number"].value_counts().head(10).index.tolist()
    )
    by_muni = {}
    for m in top_munis:
        g = test_df[test_df["municipality_number"] == m]
        by_muni[str(int(m))] = float(mean_absolute_error(g["y_true"], g["y_pred"]))

    metrics = {
        "overall": {"mae": mae, "rmse": rmse},
        "by_realestate_type_mae": by_type,
        "by_top_municipality_mae": by_muni,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    feature_schema = {
        "categorical": CATEGORICAL_COLS,
        "numeric": NUMERIC_COLS,
        "derived": ["area_ratio"],
        "label": "price",
    }

    return TrainResult(pipeline=pipeline, metrics=metrics, feature_schema=feature_schema)
