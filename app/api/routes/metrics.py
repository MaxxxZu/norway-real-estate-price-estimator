from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_registry
from app.ml.metrics_summary import build_metrics_summary
from app.ml.registry import ModelNotReadyError, ModelRegistry
from app.observability.prometheus import prometheus_response

router = APIRouter(prefix="/metrics", tags=["monitoring"])


@router.get("", summary="Get last model metrics")
def get_metrics(registry: ModelRegistry = Depends(get_registry)) -> Any:
    try:
        return registry.get_active_metrics()
    except ModelNotReadyError as e:
        raise HTTPException(
            status_code=503,
            detail={"message": "model_not_ready", "reason": str(e)},
        ) from e


@router.get("/summary")
def metrics_summary(registry: ModelRegistry = Depends(get_registry)) -> dict[str, Any]:
    try:
        raw = registry.get_active_metrics()
    except ModelNotReadyError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    return build_metrics_summary(raw)


@router.get("/prometheus")
def metrics_prometheus():
    return prometheus_response()
