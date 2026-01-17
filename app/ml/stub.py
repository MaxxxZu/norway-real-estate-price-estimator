from app.ml.base import Predictor
from app.schemas import EstimationFeatures


class StubPredictor(Predictor):
    model_version = "stub-v1"

    def predict_one(self, features: EstimationFeatures) -> int:
        # Current stub heuristic: uses both usable and total areas.
        price = int(features.usable_area * 50_000 + features.total_area * 5_000)
        return max(price, 0)
