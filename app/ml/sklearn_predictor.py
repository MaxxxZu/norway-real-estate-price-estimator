from io import BytesIO
from typing import Any

import joblib
import numpy as np

from app.features import transform_features, validate_feature_schema
from app.ml.base import Predictor
from app.schemas import EstimationFeatures


class SklearnPredictor(Predictor):
    def __init__(self, model_version: str, pipeline: Any, feature_schema: dict[str, Any]):
        validate_feature_schema(feature_schema)
        self.model_version = model_version
        self._pipeline = pipeline
        self._schema = dict(feature_schema)

    @classmethod
    def from_bytes(
        cls, model_version: str, data: bytes, feature_schema: dict[str, Any]
    ) -> "SklearnPredictor":
        # Validate before deserializing. Artifacts must still come from trusted storage.
        validate_feature_schema(feature_schema)
        return cls(model_version, joblib.load(BytesIO(data)), feature_schema)

    def predict_one(self, features: EstimationFeatures) -> int:
        df = transform_features([features.model_dump(mode="json")], self._schema)
        pred = float(self._pipeline.predict(df)[0])
        if self._schema["prediction_transform"] == "expm1":
            pred = float(np.expm1(pred))
        return max(int(pred), 0)
