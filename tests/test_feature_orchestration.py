from copy import deepcopy
from unittest.mock import Mock

import pytest

from app.config import settings
from app.training.pipeline import Pipeline
from app.training.snapshots import snapshot_paths_for_prefix
from tests.test_rolling_chain import FakeStorage
from tests.test_temporal_evaluation import config, model_spy, synthetic_rows  # noqa: F401


def configure(monkeypatch):
    for key, value in vars(config()).items():
        monkeypatch.setattr(settings, f"split_{key}", value)


def manifest():
    return {
        "period": {"end_date": "2025-12-31"},
        "counts": {"rows_trainable": 8},
        "dropped_reasons": {},
        "source_revisions": ["synthetic-source-r1"],
    }


def test_synchronous_lineage_and_blocked_promotion(monkeypatch, model_spy):  # noqa: F811
    configure(monkeypatch)
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.train = pipeline.publish = True
    pipeline.split_config = config()
    pipeline.storage = FakeStorage()
    pipeline.paths = snapshot_paths_for_prefix("synthetic")
    pipeline.manifest = manifest()
    pipeline.result = {}
    monkeypatch.setattr(
        "app.training.pipeline.load_trainable_rows_from_parquet", lambda *args: synthetic_rows()
    )
    trained = pipeline._train_model()
    assert trained.feature_schema["feature_reference_year"] == 2025
    assert trained.lineage["dataset"]["source"] == {
        "manifest_key": "synthetic/manifest.json",
        "manifest": manifest(),
    }
    pipeline._publish_model(trained)
    assert not pipeline.result["published"]["published"]
    assert "latest.json" not in pipeline.storage.writes
    key = next(k for k in pipeline.storage.writes if k.endswith("training_manifest.json"))
    saved = pipeline.storage.get_json("unused", key)
    assert saved["lineage"] == pipeline.result["lineage"]
    assert saved["comparison"]["status"] == "unverifiable"


def test_async_lineage_gate_and_manifest(monkeypatch, model_spy):  # noqa: F811
    from app.tasks import rolling

    configure(monkeypatch)
    storage = FakeStorage()
    storage.delete = Mock()
    monkeypatch.setattr(rolling, "S3Storage", lambda: storage)
    monkeypatch.setattr(rolling, "load_trainable_rows_from_parquet", lambda *a: synthetic_rows())
    ctx = {
        "dataset_key": "synthetic/dataset.parquet",
        "snapshot_prefix": "synthetic",
        "manifest_key": "synthetic/manifest.json",
        "manifest": manifest(),
    }
    rolling.train_rolling_12m.run(ctx)
    assert ctx["train"]["feature_schema"]["feature_reference_year"] == 2025
    lineage = deepcopy(ctx["train"]["lineage"])
    assert lineage["dataset"]["source"] == {
        "manifest_key": ctx["manifest_key"],
        "manifest": manifest(),
    }
    monkeypatch.setattr(settings, "train_min_rows", 0)
    rolling.gate_rolling_12m.run(ctx)
    assert ctx["gating"]["reasons"] == ["unverifiable_comparison"]
    rolling.publish_rolling_12m.run(ctx)
    assert "latest.json" not in storage.writes
    assert ctx["published"]["skipped"]
    # Exercise metadata serialization separately from the blocked gate decision.
    ctx["gating"] = {"passed": True, "reasons": []}
    rolling.publish_rolling_12m.run(ctx)
    key = next(k for k in storage.writes if k.endswith("training_manifest.json"))
    saved = storage.get_json("unused", key)
    assert saved["lineage"] == lineage
    assert saved["comparison"] == ctx["train"]["comparison"]


@pytest.mark.parametrize("mismatch", [None, "artifact", "manifest"])
def test_both_paths_save_same_paired_evaluation_and_lineage(
    monkeypatch,
    model_spy,  # noqa: F811
    mismatch,
):
    from app.features import feature_schema
    from app.tasks import rolling
    from app.training.publish import update_latest_json, upload_model_artifacts
    from app.training.splitting import split_rows
    from tests.test_temporal_evaluation import SpyModel

    configure(monkeypatch)
    monkeypatch.setattr(settings, "train_min_rows", 0)

    def seeded_storage():
        storage = FakeStorage()
        storage.delete = Mock()
        lineage = split_rows(synthetic_rows(2023, 100), config(2023)).metadata
        keys = upload_model_artifacts(
            storage,
            "champion",
            SpyModel(),
            {},
            feature_schema(2024),
            {
                "model_version": "champion",
                "snapshot_prefix": "old-synthetic",
                "period": {"start_date": "2023-01-01", "end_date": "2023-06-30"},
                "counts": {"rows_trainable": 8},
                "dropped_reasons": {},
                "metrics": {},
                "gating": {"passed": True, "reasons": [], "details": {}},
                "lineage": lineage,
            },
        )
        update_latest_json(storage, "champion", keys["model_key"], "old-synthetic")
        if mismatch == "artifact":
            pointer = storage.get_json("unused", "latest.json")
            pointer["artifact_key"] = "models/other-version/model.pkl"
            storage.put_json("unused", "latest.json", pointer)
        elif mismatch == "manifest":
            key = "models/champion/training_manifest.json"
            training_manifest = storage.get_json("unused", key)
            training_manifest["model_version"] = "other-version"
            storage.put_json("unused", key, training_manifest)
        storage.writes.clear()
        return storage

    pipeline = Pipeline.__new__(Pipeline)
    pipeline.train = pipeline.publish = True
    pipeline.split_config = config()
    champion_storage = seeded_storage()
    pipeline.storage = deepcopy(champion_storage)
    pipeline.paths = snapshot_paths_for_prefix("synthetic")
    pipeline.manifest = manifest()
    pipeline.result = {}
    monkeypatch.setattr(
        "app.training.pipeline.load_trainable_rows_from_parquet", lambda *args: synthetic_rows()
    )
    trained = pipeline._train_model()
    pipeline._publish_model(trained)
    assert pipeline.result["published"]["published"] is (mismatch is None)
    if mismatch:
        assert "latest.json" not in pipeline.storage.writes

    storage = deepcopy(champion_storage)
    monkeypatch.setattr(rolling, "S3Storage", lambda: storage)
    monkeypatch.setattr(rolling, "load_trainable_rows_from_parquet", lambda *a: synthetic_rows())
    ctx = {
        "dataset_key": "synthetic/dataset.parquet",
        "snapshot_prefix": "synthetic",
        "manifest_key": "synthetic/manifest.json",
        "manifest": manifest(),
    }
    rolling.train_rolling_12m.run(ctx, split_config=config().as_dict())
    rolling.gate_rolling_12m.run(ctx)
    rolling.publish_rolling_12m.run(ctx)
    if mismatch:
        assert ctx["published"]["skipped"]
        assert ctx["gating"]["reasons"] == ["unverifiable_comparison"]
        assert "latest.json" not in storage.writes
    else:
        assert ctx["published"]["published"]
    assert ctx["train"]["lineage"] == pipeline.result["lineage"]
    assert ctx["train"]["comparison"] == pipeline.result["comparison"]
    assert ctx["train"]["comparison"]["status"] == ("unverifiable" if mismatch else "verified")
    if mismatch:
        return
    version = ctx["published"]["model_version"]
    saved = storage.get_json("unused", f"models/{version}/training_manifest.json")
    assert saved["comparison"] == pipeline.result["comparison"]
    assert saved["lineage"] == pipeline.result["lineage"]
