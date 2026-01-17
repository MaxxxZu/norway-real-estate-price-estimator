from io import BytesIO
from typing import Any

import joblib
import pandas as pd

from app.ml.base import Predictor
from app.schemas import EstimationFeatures


class SklearnPredictor(Predictor):
    def __init__(self, model_version: str, pipeline: Any):
        self.model_version = model_version
        self._pipeline = pipeline

    @classmethod
    def from_bytes(cls, model_version: str, data: bytes) -> "SklearnPredictor":
        pipeline = joblib.load(BytesIO(data))
        return cls(model_version=model_version, pipeline=pipeline)

    def predict_one(self, features: EstimationFeatures) -> int:
        row = self._features_to_row(features)
        df = pd.DataFrame([row])
        y_pred = self._pipeline.predict(df)
        price = int(float(y_pred[0]))
        return max(price, 0)

    @staticmethod
    def _features_to_row(f: EstimationFeatures) -> dict[str, Any]:
        total_area = float(f.total_area)
        bra = float(f.bra)
        area_ratio = bra / total_area if total_area > 0 else None

        return {
            "realestate_type": str(f.realestate_type.value),
            "municipality_number": int(f.municipality_number),
            "lat": float(f.lat),
            "lon": float(f.lon),
            "built_year": int(f.built_year),
            "bra": bra,
            "total_area": total_area,
            "floor": f.floor,
            "bedrooms": f.bedrooms,
            "rooms": f.rooms,
            "area_ratio": area_ratio,
        }
