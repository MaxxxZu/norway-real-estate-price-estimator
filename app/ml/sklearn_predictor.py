from datetime import date
from io import BytesIO
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

from app.ml.base import Predictor
from app.schemas import EstimationFeatures


class SklearnPredictor(Predictor):
    def __init__(
        self,
        model_version: str,
        pipeline: Any,
        prediction_transform: Optional[str] = None,
    ):
        self.model_version = model_version
        self._pipeline = pipeline
        self._prediction_transform = prediction_transform

    @classmethod
    def from_bytes(
        cls,
        model_version: str,
        data: bytes,
        prediction_transform: Optional[str] = None,
    ) -> "SklearnPredictor":
        pipeline = joblib.load(BytesIO(data))
        return cls(
            model_version=model_version,
            pipeline=pipeline,
            prediction_transform=prediction_transform
        )

    def predict_one(self, features: EstimationFeatures) -> int:
        row = self._features_to_row(features)
        df = pd.DataFrame([row])
        y_pred = self._pipeline.predict(df)
        pred = float(y_pred[0])
        if self._prediction_transform == "expm1":
            pred = float(np.expm1(pred))

        price = int(pred)
        return max(price, 0)

    @staticmethod
    def _features_to_row(f: EstimationFeatures) -> dict[str, Any]:
        total_area = float(f.total_area)
        bra = float(f.bra)
        area_ratio = bra / total_area if total_area > 0 else None

        building_age = None
        try:
            building_age = date.today().year - int(f.built_year)
        except Exception:
            building_age = None

        return {
            "realestate_type": str(f.realestate_type.value),
            "municipality_number": int(f.municipality_number),
            "lat": float(f.lat),
            "lon": float(f.lon),
            "built_year": int(f.built_year),
            "building_age": building_age,
            "bra": bra,
            "total_area": total_area,
            "floor": f.floor,
            "bedrooms": f.bedrooms,
            "rooms": f.rooms,
            "area_ratio": area_ratio,
        }
