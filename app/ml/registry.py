import time
from dataclasses import dataclass
from typing import Optional, Any

from app.config import settings
from app.ml.base import Predictor
from app.ml.sklearn_predictor import SklearnPredictor
from app.ml.stub import StubPredictor
from app.storage.s3 import S3Storage, S3StorageError


class ModelNotReadyError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelRef:
    model_version: str
    model_type: str
    artifact_key: str


class ModelRegistry:
    """
    Loads model pointer (latest.json) and model artifact from S3-compatible storage.
    Caches predictor in-memory with TTL to avoid hitting S3 on each request.
    """

    def __init__(self, storage: S3Storage, refresh_seconds: int = 60):
        self._storage = storage
        self._refresh_seconds = max(int(refresh_seconds), 1)

        self._cached_predictor: Optional[StubPredictor] = None
        self._cached_version: Optional[str] = None
        self._cached_at: float = 0.0

    def get_predictor(self) -> Predictor:
        now = time.time()
        if self._cached_predictor and (now - self._cached_at) < self._refresh_seconds:
            return self._cached_predictor

        model_ref = self._load_latest()
        predictor = self._build_predictor(model_ref)

        self._cached_predictor = predictor
        self._cached_version = model_ref.model_version
        self._cached_at = now
        return predictor

    def _load_latest(self) -> ModelRef:
        try:
            latest = self._storage.get_json(
                bucket=settings.s3_bucket_models,
                key="latest.json",
            )
        except S3StorageError as e:
            raise ModelNotReadyError(
                "Model registry is not initialized (missing latest.json). "
                "Run training with --publish."
            ) from e

        model_version = str(latest.get("model_version", "")).strip()
        model_type = str(latest.get("type", "")).strip()
        artifact_key = str(latest.get("artifact_key", "")).strip()

        if not model_version or not model_type or not artifact_key:
            raise ModelNotReadyError("Invalid latest.json: missing required fields")

        return ModelRef(
            model_version=model_version,
            model_type=model_type,
            artifact_key=artifact_key,
        )

    def _build_predictor(self, model_ref: ModelRef) -> Predictor:
        if model_ref.model_type not in {"stub", "sklearn"}:
            raise ModelNotReadyError(f"Unsupported model type: {model_ref.model_type}")

        if model_ref.model_type == "stub":
            artifact = self._storage.get_json(
                bucket=settings.s3_bucket_models,
                key=model_ref.artifact_key
            )
            return StubPredictor.from_artifact(
                model_version=model_ref.model_version,
                artifact=artifact
            )

        if model_ref.model_type == "sklearn":
            data = self._storage.get_bytes(
                bucket=settings.s3_bucket_models,
                key=model_ref.artifact_key
            )
            return SklearnPredictor.from_bytes(model_version=model_ref.model_version, data=data)

    def get_active_metrics(self) -> dict[str, Any]:
        ref = self._load_latest()
        prefix = ref.artifact_key.rsplit("/", 1)[0]
        metrics_key = f"{prefix}/metrics.json"

        try:
            metrics = self._storage.get_json(
                bucket=settings.s3_bucket_models,
                key=metrics_key,
            )
        except S3StorageError as e:
            raise ModelNotReadyError(
                f"Metrics not found for active model ({metrics_key})"
            ) from e

        return {
            "model_version": ref.model_version,
            "model_type": ref.model_type,
            "metrics_key": metrics_key,
            "metrics": metrics,
        }
