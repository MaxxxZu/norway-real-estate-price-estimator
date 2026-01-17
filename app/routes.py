from typing import Any

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.schemas import EstimateRequest, EstimateResponse, EstimateResult

router = APIRouter()


@router.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "service": "ree", "env": settings.env}


@router.post("/estimate", response_model=EstimateResponse)
def estimate(payload: EstimateRequest) -> Any:
    """
    Batch estimation endpoint.
    Payload format: { "<property_id>": { ...features... }, ... }

    Strict contract:
      - bra and total_area are required
      - realestate_type (enum key), built_year, municipality_number, lat/lon required
      - floor required for leilighet
    """
    if not payload:
        raise HTTPException(status_code=422, detail="Empty payload")

    results: dict[str, EstimateResult] = {}

    for property_id, f in payload.items():
        # Stub heuristic using both areas + light geo/type adjustments later.
        # Replace with model inference later.
        estimated = max(int(f.usable_area * 50_000 + f.total_area * 5_000), 0)

        warnings: list[str] = []
        # We can keep some soft warnings for non-required optional fields
        if f.bedrooms is None:
            warnings.append("bedrooms_missing")
        if f.rooms is None:
            warnings.append("rooms_missing")

        results[property_id] = EstimateResult(
            estimated_price=estimated,
            currency="NOK",
            model_version="stub-v1",
            warnings=warnings,
        )

    return results
