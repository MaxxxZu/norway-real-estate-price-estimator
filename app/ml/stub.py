from typing import Any

from app.ml.base import Predictor
from app.schemas import EstimationFeatures


class StubPredictor(Predictor):
    def __init__(self, model_version: str, usable_area_coef: float, total_area_coef: float):
        self.model_version = model_version
        self.usable_area_coef = float(usable_area_coef)
        self.total_area_coef = float(total_area_coef)

    @classmethod
    def from_artifact(cls, model_version: str, artifact: dict[str, Any]) -> "StubPredictor":
        params = artifact.get("params", {})
        usable_area_coef = params.get("usable_area_coef", 50_000)
        total_area_coef = params.get("total_area_coef", 5_000)
        return cls(
            model_version=model_version,
            usable_area_coef=usable_area_coef,
            total_area_coef=total_area_coef,
        )

    def predict_one(self, features: EstimationFeatures) -> int:
        usable_area = features.usable_area * self.usable_area_coef
        total_area = features.total_area * self.total_area_coef
        price = int(usable_area + total_area)

        return max(price, 0)
