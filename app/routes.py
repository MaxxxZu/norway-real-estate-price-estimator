from typing import Any

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.ml.stub import StubPredictor
from app.schemas import EstimateResponse, ValidationErrorResponse
from app.services.estimate_service import estimate_batch

router = APIRouter()
predictor = StubPredictor()


@router.get("/healthz", tags=["health"])
def healthz() -> dict:
    return {"status": "ok", "service": "ree", "env": settings.env}


@router.post(
    "/estimate",
    response_model=EstimateResponse,
    responses={422: {"model": ValidationErrorResponse}},
    tags=["estimation"],
)
def estimate(payload: dict[str, dict[str, Any]]) -> Any:
    if not payload:
        raise HTTPException(status_code=422, detail="Empty payload")

    results, errors = estimate_batch(payload, predictor=predictor)
    if errors:
        raise HTTPException(
            status_code=422,
            detail={"message": "validation_failed", "errors": errors}
        )

    return results
