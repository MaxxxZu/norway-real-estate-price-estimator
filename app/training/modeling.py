from dataclasses import dataclass
from typing import Any
from datetime import date

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor

from app.training.metrics import compute_metrics


CATEGORICAL_COLS: list[str] = [
    "realestate_type",
    "municipality_number",
]
NUMERIC_COLS = [
    "lat",
    "lon",
    "built_year",
    "building_age",
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


def _metrics_by_group(df: pd.DataFrame, group_col: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for key, g in df.groupby(group_col, dropna=False):
        out[str(key)] = compute_metrics(
            g["y_true"].to_numpy(dtype=float),
            g["y_pred"].to_numpy(dtype=float),
        )
    return out


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    bra = pd.to_numeric(df.get("bra"), errors="coerce")
    total_area = pd.to_numeric(df.get("total_area"), errors="coerce")
    df["area_ratio"] = np.where((total_area > 0) & bra.notna(), bra / total_area, np.nan)

    current_year = date.today().year
    built_year = pd.to_numeric(df.get("built_year"), errors="coerce")
    df["building_age"] = (current_year - built_year).where(built_year.notna(), np.nan)

    return df


def train_and_evaluate(rows: list[dict[str, Any]]) -> TrainResult:
    df = pd.DataFrame(rows)
    df = _add_derived_features(df)

    y = df["price"].astype(float)
    y_log = np.log1p(y)

    X = df[CATEGORICAL_COLS + NUMERIC_COLS].copy()

    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, CATEGORICAL_COLS),
            ("num", numeric_transformer, NUMERIC_COLS),
        ],
        sparse_threshold=0.0,
    )

    model = HistGradientBoostingRegressor(
        random_state=42,
        learning_rate=0.05,
        max_iter=600,
        max_depth=8,
        min_samples_leaf=20,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train_log)

    y_pred_log = pipeline.predict(X_test)

    y_true = np.expm1(y_test_log.to_numpy(dtype=float))
    y_pred = np.expm1(y_pred_log.astype(float))

    eval_df = X_test.copy()
    eval_df["y_true"] = y_true
    eval_df["y_pred"] = y_pred
    eval_df["realestate_type"] = X_test["realestate_type"].astype(str).values

    overall = compute_metrics(eval_df["y_true"].to_numpy(), eval_df["y_pred"].to_numpy())
    by_type = _metrics_by_group(eval_df, "realestate_type")

    metrics = {
        "overall": overall,
        "by_realestate_type": by_type,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "target_transform": "log1p",
        "prediction_transform": "expm1",
        "model_family": "HistGradientBoostingRegressor",
    }

    feature_schema = {
        "categorical": CATEGORICAL_COLS,
        "numeric": NUMERIC_COLS,
        "derived": ["area_ratio", "building_age"],
        "label": "price",
        "target_transform": "log1p",
        "prediction_transform": "expm1",
    }

    return TrainResult(pipeline=pipeline, metrics=metrics, feature_schema=feature_schema)
