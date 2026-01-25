from typing import Any

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.ml.metrics_summary import build_metrics_summary
from app.ml.registry import ModelNotReadyError, ModelRegistry
from app.observability.prometheus import prometheus_response
from app.storage.s3 import S3Storage

router = APIRouter(prefix="/metrics", tags=["monitoring"])

_storage = S3Storage()
_registry = ModelRegistry(_storage, refresh_seconds=settings.model_registry_refresh_seconds)


@router.get("", summary="Get last model metrics")
def get_metrics() -> Any:
    try:
        return _registry.get_active_metrics()
    except ModelNotReadyError as e:
        raise HTTPException(
            status_code=503,
            detail={"message": "model_not_ready", "reason": str(e)},
        )


@router.get("/summary")
def metrics_summary() -> dict[str, Any]:
    try:
        raw = _registry.get_active_metrics()
    except ModelNotReadyError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return build_metrics_summary(raw)


@router.get("/prometheus")
def metrics_prometheus():
    return prometheus_response()
