from dataclasses import dataclass
from typing import Any

import numpy as np
from catboost import CatBoostRegressor, Pool

from app.features import CATEGORICAL_COLS, feature_schema, transform_features
from app.training.metrics import compute_metrics
from app.training.splitting import TemporalSplit, TemporalSplitConfig, split_rows


@dataclass(frozen=True)
class TrainResult:
    pipeline: Any
    metrics: dict[str, Any]
    feature_schema: dict[str, Any]
    lineage: dict[str, Any]


def evaluate_model(model, rows, schema):
    """Evaluate every row in caller order with this model's saved feature contract."""
    features = transform_features(rows, schema)
    predictions = np.expm1(np.asarray(model.predict(features), dtype=float))
    return evaluate_predictions(rows, predictions)


def evaluate_predictions(rows, predictions):
    predictions = np.asarray(predictions, dtype=float)
    if predictions.shape != (len(rows),) or not np.isfinite(predictions).all():
        raise ValueError("Expected one finite prediction per evaluation row")
    labels = np.array([row["price"] for row in rows], dtype=float)
    segments = np.array([row["realestate_type"] for row in rows])
    return {
        "overall": compute_metrics(labels, predictions),
        "by_realestate_type": {
            segment: compute_metrics(labels[segments == segment], predictions[segments == segment])
            for segment in sorted(set(segments))
        },
    }


def fit_model(split: TemporalSplit, schema):
    train = transform_features(split.train, schema)
    validation = transform_features(split.validation, schema)
    if not any(train[column].nunique(dropna=False) > 1 for column in train.columns):
        raise ValueError("CatBoost requires at least one varying train feature")
    categorical = [train.columns.get_loc(col) for col in CATEGORICAL_COLS]
    train_pool = Pool(
        train, np.log1p([row["price"] for row in split.train]), cat_features=categorical
    )
    validation_pool = Pool(
        validation, np.log1p([row["price"] for row in split.validation]), cat_features=categorical
    )
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
    model.fit(train_pool, eval_set=validation_pool, use_best_model=True)
    return model


def baseline_metrics(split):
    global_median = float(np.median([row["price"] for row in split.train]))
    medians = {
        segment: float(
            np.median([row["price"] for row in split.train if row["realestate_type"] == segment])
        )
        for segment in {row["realestate_type"] for row in split.train}
    }
    predictions = [medians.get(row["realestate_type"], global_median) for row in split.test]
    return {
        "kind": "train_segment_median",
        "global_median": global_median,
        "segment_medians": medians,
        **evaluate_predictions(split.test, predictions),
    }


def train_and_evaluate(
    rows, *, reference_year: int, split_config: TemporalSplitConfig, dataset_identity=None
) -> TrainResult:
    split = split_rows(rows, split_config, dataset_identity)
    schema = feature_schema(reference_year)
    model = fit_model(split, schema)
    metrics = {
        **evaluate_model(model, split.test, schema),
        "n_train": len(split.train),
        "n_validation": len(split.validation),
        "n_test": len(split.test),
        "target_transform": "log1p",
        "prediction_transform": "expm1",
        "model_family": "CatBoostRegressor",
        "evaluation": split.metadata["partitions"]["test"],
        "baseline": baseline_metrics(split),
    }
    return TrainResult(model, metrics, schema, split.metadata)
