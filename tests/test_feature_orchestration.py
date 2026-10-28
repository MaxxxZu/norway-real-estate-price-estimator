from types import SimpleNamespace
from unittest.mock import Mock

from app.training.pipeline import Pipeline
from app.training.snapshots import snapshot_paths_for_prefix


def test_synchronous_training_uses_snapshot_period_reference_year(monkeypatch):
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.train = True
    pipeline.storage = object()
    pipeline.paths = snapshot_paths_for_prefix("synthetic")
    pipeline.manifest = {"period": {"end_date": "2025-12-31"}}
    pipeline.result = {}
    monkeypatch.setattr(
        "app.training.pipeline.load_trainable_rows_from_parquet", lambda *args: ["synthetic-rows"]
    )
    trainer = Mock(return_value=SimpleNamespace(metrics={}))
    monkeypatch.setattr("app.training.pipeline.train_and_evaluate", trainer)
    pipeline._train_model()
    trainer.assert_called_once_with(["synthetic-rows"], reference_year=2025)


def test_async_training_uses_snapshot_period_reference_year(monkeypatch):
    from app.tasks import rolling

    storage = Mock()
    monkeypatch.setattr(rolling, "S3Storage", lambda: storage)
    monkeypatch.setattr(rolling, "load_trainable_rows_from_parquet", lambda *a: ["synthetic"])
    trainer = Mock(
        return_value=SimpleNamespace(pipeline="model-spy", metrics={}, feature_schema={})
    )
    monkeypatch.setattr(rolling, "train_and_evaluate", trainer)
    ctx = {
        "dataset_key": "synthetic/dataset.parquet",
        "snapshot_prefix": "synthetic",
        "manifest": {"period": {"end_date": "2025-12-31"}},
    }
    rolling.train_rolling_12m.run(ctx)
    trainer.assert_called_once_with(["synthetic"], reference_year=2025)
    storage.put_bytes.assert_called_once()
