"""Shared versioned feature contract for training and serving."""

from typing import Any

import pandas as pd

CATEGORICAL_COLS = ["realestate_type", "municipality_number"]
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
DERIVED_COLS = ["area_ratio", "building_age"]


def feature_schema(reference_year: int) -> dict[str, Any]:
    if type(reference_year) is not int or not 1800 <= reference_year <= 9999:
        raise ValueError("Invalid feature reference year")
    return {
        "schema_version": 1,
        "feature_reference_year": reference_year,
        "categorical": list(CATEGORICAL_COLS),
        "numeric": list(NUMERIC_COLS),
        "derived": list(DERIVED_COLS),
        "columns": CATEGORICAL_COLS + NUMERIC_COLS,
        "categorical_dtype": "object[str]",
        "numeric_dtype": "float64",
        "missing_numeric": "NaN",
        "label": "price",
        "target_transform": "log1p",
        "prediction_transform": "expm1",
    }


def validate_feature_schema(schema: dict[str, Any]) -> None:
    if not isinstance(schema, dict) or type(schema.get("schema_version")) is not int:
        raise ValueError("Missing or invalid feature schema version")
    expected = feature_schema(schema.get("feature_reference_year"))
    if schema != expected:
        raise ValueError(
            "Unsupported feature schema; explicit retraining/compatibility decision required"
        )


def transform_features(rows: list[dict[str, Any]], schema: dict[str, Any]) -> pd.DataFrame:
    validate_feature_schema(schema)
    df = pd.DataFrame(rows)
    for col in NUMERIC_COLS:
        if col not in DERIVED_COLS:
            df[col] = pd.to_numeric(
                df.get(col, pd.Series(index=df.index, dtype=float)), errors="raise"
            ).astype("float64")
    df["area_ratio"] = df["bra"] / df["total_area"]
    df["building_age"] = schema["feature_reference_year"] - df["built_year"]
    for col in CATEGORICAL_COLS:
        df[col] = df[col].map(lambda value: str(getattr(value, "value", value))).astype(object)
    return df[CATEGORICAL_COLS + NUMERIC_COLS]
