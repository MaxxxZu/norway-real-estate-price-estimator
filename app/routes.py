from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException

from app.config import settings
from app.ml.registry import ModelNotReadyError, ModelRegistry
from app.schemas import EstimateResponse, ValidationErrorResponse
from app.services.estimate_service import estimate_batch
from app.storage.s3 import S3Storage
from app.ml.base import Predictor

from app.api.examples import ESTIMATE_REQUEST_EXAMPLES


router = APIRouter()

_storage = S3Storage()
_registry = ModelRegistry(_storage, refresh_seconds=settings.model_registry_refresh_seconds)


def get_predictor() -> Predictor:
    try:
        return _registry.get_predictor()
    except ModelNotReadyError as e:
        raise HTTPException(
            status_code=503,
            detail={"message": "model_not_ready", "reason": str(e)}
        )


@router.get("/healthz", tags=["health"])
def healthz() -> dict:
    return {"status": "ok", "service": "ree", "env": settings.env}


@router.post(
    "/estimate",
    response_model=EstimateResponse,
    responses={
        422: {"model": ValidationErrorResponse},
        503: {"description": "Model not ready"},
    },
    tags=["estimation"],
)
def estimate(
    payload: dict[str, dict[str, Any]] = Body(
        ..., openapi_examples=ESTIMATE_REQUEST_EXAMPLES
    ),
    predictor: Predictor = Depends(get_predictor),
) -> Any:
    if not payload:
        raise HTTPException(status_code=422, detail="Empty payload")

    results, errors = estimate_batch(payload, predictor=predictor)
    if errors:
        raise HTTPException(
            status_code=422,
            detail={"message": "validation_failed", "errors": errors}
        )

    return results
