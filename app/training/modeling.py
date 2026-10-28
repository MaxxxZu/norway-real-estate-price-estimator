from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split

from app.features import CATEGORICAL_COLS, feature_schema, transform_features
from app.training.dataset import build_trainable_dataset
from app.training.metrics import compute_metrics

TARGET_TRANSFORM = "log1p"
PREDICTION_TRANSFORM = "expm1"
MODEL_FAMILY = "CatBoostRegressor"


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


def train_and_evaluate(rows: list[dict[str, Any]], *, reference_year: int) -> TrainResult:
    result = build_trainable_dataset(rows)
    if result.dropped_reasons or not result.trainable_rows:
        raise ValueError(f"Training requires normalized valid rows: {result.dropped_reasons}")
    rows = result.trainable_rows
    schema = feature_schema(reference_year)
    X = transform_features(rows, schema)
    y_log = np.log1p(pd.Series([row["price"] for row in rows], dtype=float))

    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.2, random_state=42, stratify=X["realestate_type"]
    )
    cat_features = [X.columns.get_loc(c) for c in CATEGORICAL_COLS]
    train_pool = Pool(X_train, y_train_log, cat_features=cat_features)
    test_pool = Pool(X_test, y_test_log, cat_features=cat_features)

    model = CatBoostRegressor(
        loss_function="RMSE",
        learning_rate=0.05,
        depth=8,
        iterations=5000,
        random_seed=42,
        eval_metric="RMSE",
        od_type="Iter",
        od_wait=200,
        verbose=False,
    )

    model.fit(train_pool, eval_set=test_pool, use_best_model=True)

    y_pred_log = model.predict(X_test)
    y_true = np.expm1(y_test_log.to_numpy(dtype=float))
    y_pred = np.expm1(np.array(y_pred_log, dtype=float))

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
        "target_transform": TARGET_TRANSFORM,
        "prediction_transform": PREDICTION_TRANSFORM,
        "model_family": MODEL_FAMILY,
    }

    return TrainResult(pipeline=model, metrics=metrics, feature_schema=schema)
