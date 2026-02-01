from datetime import UTC, datetime
from io import BytesIO
from typing import Any

import joblib

from app.config import settings
from app.storage.s3 import S3Storage, S3StorageError


def try_load_previous_metrics(storage: S3Storage) -> dict[str, Any] | None:
    """
    Load metrics.json of the currently published model (pointed by latest.json).
    If latest.json is missing or malformed, return None.
    """
    try:
        latest = storage.get_json(bucket=settings.s3_bucket_models, key="latest.json")
    except S3StorageError:
        return None

    try:
        model_type = str(latest.get("type", "")).strip()
        model_version = str(latest.get("model_version", "")).strip()
        if not model_type or not model_version:
            return None

        metrics_key = f"models/{model_version}/metrics.json"
        metrics = storage.get_json(bucket=settings.s3_bucket_models, key=metrics_key)
        return metrics if isinstance(metrics, dict) else None
    except Exception:
        return None


def _upload_model_artifacts_from_bytes(
    storage: S3Storage,
    model_version: str,
    pipeline_bytes: bytes,
    metrics: dict[str, Any],
    feature_schema: dict[str, Any],
    training_manifest: dict[str, Any],
) -> dict[str, str]:
    prefix = f"models/{model_version}"
    model_key = f"{prefix}/model.pkl"
    metrics_key = f"{prefix}/metrics.json"
    schema_key = f"{prefix}/feature_schema.json"
    manifest_key = f"{prefix}/training_manifest.json"

    storage.put_bytes(
        bucket=settings.s3_bucket_models,
        key=model_key,
        data=pipeline_bytes,
        content_type="application/octet-stream",
    )

    storage.put_json(bucket=settings.s3_bucket_models, key=metrics_key, obj=metrics)
    storage.put_json(bucket=settings.s3_bucket_models, key=schema_key, obj=feature_schema)
    storage.put_json(bucket=settings.s3_bucket_models, key=manifest_key, obj=training_manifest)

    return {
        "model_key": model_key,
        "metrics_key": metrics_key,
        "schema_key": schema_key,
        "manifest_key": manifest_key,
    }


def upload_model_artifacts(
    storage: S3Storage,
    model_version: str,
    pipeline: Any,
    metrics: dict[str, Any],
    feature_schema: dict[str, Any],
    training_manifest: dict[str, Any],
) -> dict[str, str]:
    bio = BytesIO()
    joblib.dump(pipeline, bio)
    return _upload_model_artifacts_from_bytes(
        storage=storage,
        model_version=model_version,
        pipeline_bytes=bio.getvalue(),
        metrics=metrics,
        feature_schema=feature_schema,
        training_manifest=training_manifest,
    )


def upload_model_artifacts_from_bytes(
    storage: S3Storage,
    model_version: str,
    pipeline_bytes: bytes,
    metrics: dict[str, Any],
    feature_schema: dict[str, Any],
    training_manifest: dict[str, Any],
) -> dict[str, str]:
    return _upload_model_artifacts_from_bytes(
        storage=storage,
        model_version=model_version,
        pipeline_bytes=pipeline_bytes,
        metrics=metrics,
        feature_schema=feature_schema,
        training_manifest=training_manifest,
    )


def update_latest_json(
    storage: S3Storage,
    model_version: str,
    artifact_key: str,
    snapshot_prefix: str,
) -> None:
    latest = {
        "model_version": model_version,
        "type": "sklearn",
        "artifact_key": artifact_key,
        "created_at": datetime.now(UTC).isoformat(),
        "snapshot_prefix": snapshot_prefix,
    }
    storage.put_json(bucket=settings.s3_bucket_models, key="latest.json", obj=latest)
