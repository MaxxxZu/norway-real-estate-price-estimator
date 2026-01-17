from abc import ABC, abstractmethod

from app.schemas import EstimationFeatures


class Predictor(ABC):
    @abstractmethod
    def predict_one(self, features: EstimationFeatures) -> int:
        raise NotImplementedError
