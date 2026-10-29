import datetime
from io import BytesIO
from unittest.mock import Mock

import joblib
import numpy as np
import pandas as pd
import pytest

from app.features import feature_schema, transform_features, validate_feature_schema
from app.ml.registry import ModelNotReadyError, ModelRegistry
from app.ml.sklearn_predictor import SklearnPredictor
from app.schemas import EstimationFeatures
from tests.test_rolling_chain import row
from tests.test_temporal_evaluation import config, synthetic_rows


class SpyModel:
    def __init__(self, **kwargs):
        self.frames = []

    def fit(self, pool, **kwargs):
        self.train_frame = pool

    def predict(self, frame):
        self.frames.append(frame.copy())
        return np.repeat(np.log1p(1000000), len(frame))


def test_training_and_inference_feature_parity_without_fit(monkeypatch):
    import app.training.modeling as modeling

    monkeypatch.setattr(modeling, "CatBoostRegressor", SpyModel)
    monkeypatch.setattr(modeling, "Pool", lambda data, *args, **kwargs: data)
    rows = synthetic_rows()
    trained = modeling.train_and_evaluate(rows, reference_year=2025, split_config=config())
    predictor = SklearnPredictor("v1", trained.pipeline, trained.feature_schema)
    predictor.predict_one(EstimationFeatures.model_validate(rows[0]))
    training_frame = trained.pipeline.train_frame.reset_index(drop=True)
    inference_frame = trained.pipeline.frames[-1]
    pd.testing.assert_frame_equal(training_frame.iloc[:1], inference_frame)
    assert inference_frame["municipality_number"].iloc[0] == "301"
    assert inference_frame["building_age"].iloc[0] == 45
    assert inference_frame["rooms"].dtype == np.dtype("float64")
    assert inference_frame["rooms"].iloc[0] == 2


def test_saved_model_schema_independent_of_system_date(monkeypatch):
    schema = feature_schema(2025)
    serialized = BytesIO()
    joblib.dump(SpyModel(), serialized)
    frames = []
    real_date = datetime.date
    for year in (2026, 2035):

        class FrozenDate(real_date):
            @classmethod
            def today(cls, frozen_year=year):
                return cls(frozen_year, 1, 1)

        monkeypatch.setattr(datetime, "date", FrozenDate)
        predictor = SklearnPredictor.from_bytes("v1", serialized.getvalue(), schema)
        predictor.predict_one(EstimationFeatures.model_validate(row()))
        frames.append(predictor._pipeline.frames[-1])
    pd.testing.assert_frame_equal(*frames)
    assert frames[0]["building_age"].iloc[0] == 25


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": 2},
        {"schema_version": True},
        {"feature_reference_year": None},
        {"feature_reference_year": True},
        {"numeric": []},
        {"prediction_transform": "log1p"},
        {"columns": []},
    ],
)
def test_incompatible_schemas_rejected(change):
    with pytest.raises(ValueError):
        validate_feature_schema({**feature_schema(2025), **change})


def test_legacy_model_rejected_before_deserialization(monkeypatch):
    loader = Mock()
    monkeypatch.setattr(joblib, "load", loader)
    with pytest.raises(ValueError):
        SklearnPredictor.from_bytes("old", b"unused", {"prediction_transform": "expm1"})
    storage = Mock()
    storage.get_json.side_effect = [
        {"model_version": "old", "type": "sklearn", "artifact_key": "old/model.pkl"},
        {"prediction_transform": "expm1"},
    ]
    storage.get_bytes.return_value = b"unused"
    with pytest.raises(ModelNotReadyError, match="legacy"):
        ModelRegistry(storage).get_predictor()
    loader.assert_not_called()


def test_derived_values_overwritten_and_schema_order_preserved():
    expected = transform_features([row()], feature_schema(2025))
    actual = transform_features([row(building_age=999, area_ratio=999)], feature_schema(2025))
    pd.testing.assert_frame_equal(expected, actual)
    assert list(actual.columns) == feature_schema(2025)["columns"]


def test_schema_and_reference_year_roundtrip_through_artifact_writer_and_registry():
    from app.training.publish import update_latest_json, upload_model_artifacts
    from tests.test_rolling_chain import FakeStorage

    storage = FakeStorage()
    schema = feature_schema(2025)
    keys = upload_model_artifacts(storage, "v1", SpyModel(), {}, schema, {})
    update_latest_json(storage, "v1", keys["model_key"], "synthetic-snapshot")
    predictor = ModelRegistry(storage).get_predictor()
    # Existing integer truncation may lose one NOK after a floating log round trip.
    assert abs(predictor.predict_one(EstimationFeatures.model_validate(row())) - 1000000) <= 1
    assert predictor._schema == schema
    assert predictor._pipeline.frames[-1]["building_age"].iloc[0] == 25
