from typing import Any

from pydantic import ValidationError

from app.ml.base import Predictor
from app.schemas import EstimationFeatures, EstimateResponse, EstimateResult


def _format_validation_error(err: ValidationError) -> list[str]:
    msgs: list[str] = []
    for e in err.errors():
        loc = ".".join(str(x) for x in e.get("loc", []) if x != "__root__")
        typ = e.get("type", "")
        msg = e.get("msg", "invalid")
        if loc:
            msgs.append(f"{loc}: {msg} ({typ})".strip())
        else:
            msgs.append(f"{msg} ({typ})".strip())
    return msgs or ["invalid payload"]


def estimate_batch(
    payload: dict[str, dict[str, Any]],
    predictor: Predictor,
) -> tuple[EstimateResponse, dict[str, list[str]]]:
    errors: dict[str, list[str]] = {}
    features_by_id: dict[str, EstimationFeatures] = {}

    for property_id, raw in payload.items():
        try:
            features_by_id[property_id] = EstimationFeatures.model_validate(raw)
        except ValidationError as e:
            errors[property_id] = _format_validation_error(e)

    if errors:
        return {}, errors

    results: dict[str, EstimateResult] = {}
    for property_id, f in features_by_id.items():
        estimated = predictor.predict_one(f)

        warnings: list[str] = []
        if f.bedrooms is None:
            warnings.append("bedrooms_missing")
        if f.rooms is None:
            warnings.append("rooms_missing")

        results[property_id] = EstimateResult(
            estimated_price=estimated,
            currency="NOK",
            model_version=getattr(predictor, "model_version", "unknown"),
            warnings=warnings,
        )

    return results, {}
