from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, status

from app.api.examples import ESTIMATE_REQUEST_EXAMPLES
from app.api.routes.metrics import router as metrics_router
from app.config import settings
from app.dependencies import get_predictor
from app.ml.base import Predictor
from app.schemas import EstimateRequest, EstimateResponse, HealthCheckResponse
from app.services.estimate_service import estimate_batch

router = APIRouter()
router.include_router(metrics_router)


@router.get(
    "/health",
    tags=["healthcheck"],
    summary="Perform a Health Check",
    response_description="Return HTTP Status Code 200 (OK)",
    status_code=status.HTTP_200_OK,
    response_model=HealthCheckResponse,
)
async def get_health() -> HealthCheckResponse:
    return HealthCheckResponse(status="ok", env=settings.env)


@router.post(
    "/estimate",
    response_model=EstimateResponse,
    responses={
        503: {"description": "Model not ready"},
    },
    tags=["estimation"],
)
def estimate(
    payload: EstimateRequest = Body(..., openapi_examples=ESTIMATE_REQUEST_EXAMPLES),
    predictor: Predictor = Depends(get_predictor),
) -> Any:
    if not payload:
        raise HTTPException(status_code=422, detail="Empty payload")

    return estimate_batch(payload, predictor=predictor)


@router.get(
    "/ready",
    tags=["readycheck"],
    summary="Perform a Ready Check",
    response_description="Return HTTP Status Code 200 (OK)",
    status_code=status.HTTP_200_OK,
)
def get_ready(predictor: Predictor = Depends(get_predictor)) -> dict[str, str]:
    return {"status": "ready"}
