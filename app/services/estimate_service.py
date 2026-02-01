from app.ml.base import Predictor
from app.schemas import EstimateResponse, EstimateResult, EstimationFeatures


def estimate_batch(
    payload: dict[str, EstimationFeatures],
    predictor: Predictor,
) -> EstimateResponse:
    features_by_id = payload

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

    return results
