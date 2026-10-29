"""Synthetic contracts only: no CatBoost fitting or external storage."""

from copy import deepcopy
from datetime import date
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from app.features import feature_schema
from app.training.comparison import compare_models, compare_with_champion, validate_lineage
from app.training.modeling import baseline_metrics, train_and_evaluate
from app.training.splitting import TemporalSplitConfig, split_rows
from tests.test_rolling_chain import row


def config(year=2025):
    return TemporalSplitConfig(*(date(year, month, 1) for month in (1, 3, 5, 7)))


def synthetic_rows(year=2025, offset=0):
    return [
        row(
            id=offset + i + 1,
            turnover_date=f"{year}-{month:02d}-{day:02d}",
            price=1_000_000 + i * 130_000,
            built_year=1980 + i,
            bra=50 + i,
            total_area=70 + i * 2,
            rooms=2 + i % 4,
            municipality_number=301 + i,
            lat=59.1 + i / 100,
            lon=10.1 + i / 100,
            realestate_type=segment,
        )
        for i, (month, day, segment) in enumerate(
            [
                (1, 1, "enebolig"),
                (2, 10, "rekkehus"),
                (2, 10, "enebolig"),
                (3, 1, "enebolig"),
                (4, 30, "rekkehus"),
                (5, 1, "enebolig"),
                (5, 1, "hytte"),
                (6, 30, "rekkehus"),
            ]
        )
    ]


class SpyModel:
    def __init__(self, **kwargs):
        self.frames = []
        self.fit_calls = []

    def fit(self, pool, **kwargs):
        self.fit_calls.append((pool, kwargs))
        self.train_frame = pool[0]

    def predict(self, frame):
        self.frames.append(frame.copy())
        return np.repeat(np.log1p(1_000_000), len(frame))


@pytest.fixture
def model_spy(monkeypatch):
    import app.training.modeling as modeling

    monkeypatch.setattr(modeling, "CatBoostRegressor", SpyModel)
    monkeypatch.setattr(modeling, "Pool", lambda data, labels, **kw: (data, labels, kw))


def test_temporal_order_boundaries_counts_and_permutation():
    rows = synthetic_rows()
    split = split_rows(rows, config(), {"revision": "synthetic-r1"})
    shuffled = split_rows(rows[::-1], config(), {"revision": "synthetic-r1"})
    assert split == shuffled
    assert [len(split.train), len(split.validation), len(split.test)] == [3, 2, 3]
    assert max(r["turnover_date"] for r in split.train) < min(
        r["turnover_date"] for r in split.validation
    )
    assert max(r["turnover_date"] for r in split.validation) < min(
        r["turnover_date"] for r in split.test
    )
    assert split.metadata["partitions"]["test"]["segments"]["hytte"] == 1
    assert "hytte" not in split.metadata["partitions"]["train"]["segments"]
    validate_lineage(split.metadata)
    changed = deepcopy(rows)
    changed[0]["bra"] += 1
    assert (
        split_rows(changed, config()).metadata["dataset"]["fingerprint"]
        != split.metadata["dataset"]["fingerprint"]
    )


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda r: r[:3], "nonempty"),
        (lambda r: r[3:], "nonempty"),
        (lambda r: r[:3] + r[5:], "nonempty"),
        (lambda r: [dict(x, price=10) if i < 3 else x for i, x in enumerate(r)], "distinct"),
        (lambda r: [dict(x, id=1) if i == 4 else x for i, x in enumerate(r)], "overlap"),
        (
            lambda r: [
                dict(x, turnover_date="2025-07-01") if i == 7 else x for i, x in enumerate(r)
            ],
            "outside",
        ),
        (
            lambda r: [
                dict(x, turnover_date="2024-12-31") if i == 0 else x for i, x in enumerate(r)
            ],
            "outside",
        ),
        (
            lambda r: [dict(x, price=float("nan")) if i == 0 else x for i, x in enumerate(r)],
            "valid rows",
        ),
    ],
)
def test_infeasible_rejected_before_model_creation(monkeypatch, change, message):
    factory = Mock()
    monkeypatch.setattr("app.training.modeling.CatBoostRegressor", factory)
    with pytest.raises(ValueError, match=message):
        train_and_evaluate(change(synthetic_rows()), reference_year=2025, split_config=config())
    factory.assert_not_called()


@pytest.mark.parametrize(
    "values",
    [
        (None, None, None, None),
        (date(2025, 1, 1),) * 4,
        (date(2025, 1, 1), date(2025, 5, 1), date(2025, 3, 1), date(2025, 7, 1)),
    ],
)
def test_invalid_boundaries(values):
    with pytest.raises(ValueError, match="Require"):
        TemporalSplitConfig(*values)


def test_fit_sees_only_train_and_validation_and_stable_order(model_spy):
    result = train_and_evaluate(synthetic_rows()[::-1], reference_year=2025, split_config=config())
    split = split_rows(synthetic_rows(), config())
    train_pool, kwargs = result.pipeline.fit_calls[0]
    validation_pool = kwargs["eval_set"]
    assert set(train_pool[0]["bra"]) == {r["bra"] for r in split.train}
    assert set(validation_pool[0]["bra"]) == {r["bra"] for r in split.validation}
    assert set(result.pipeline.frames[0]["bra"]) == {r["bra"] for r in split.test}
    np.testing.assert_allclose(train_pool[1], np.log1p([r["price"] for r in split.train]))
    np.testing.assert_allclose(
        validation_pool[1], np.log1p([r["price"] for r in split.validation])
    )
    assert kwargs["use_best_model"] is True
    assert result.metrics["evaluation"] == result.lineage["partitions"]["test"]


def test_smallest_useful_split_and_repeated_property_within_train(model_spy):
    rows = synthetic_rows()
    rows[1]["id"] = rows[0]["id"]
    result = train_and_evaluate(
        [rows[i] for i in (0, 1, 3, 5)], reference_year=2025, split_config=config()
    )
    assert result.metrics["n_train"] == 2
    assert result.metrics["n_validation"] == result.metrics["n_test"] == 1


def test_baseline_train_only_and_fallback():
    rows = synthetic_rows()
    split = split_rows(rows, config())
    baseline = baseline_metrics(split)
    assert baseline["segment_medians"]["enebolig"] == 1_130_000
    assert baseline["global_median"] == 1_130_000
    assert "hytte" not in baseline["segment_medians"]
    altered = [dict(r, price=r["price"] * 50) if i >= 3 else r for i, r in enumerate(rows)]
    other = baseline_metrics(split_rows(altered, config()))
    assert other["segment_medians"] == baseline["segment_medians"]
    assert other["global_median"] == baseline["global_median"]
    expected = np.array([1_130_000, 1_130_000, 1_130_000])
    assert baseline["overall"]["mae"] == np.mean(
        np.abs(np.array([r["price"] for r in split.test]) - expected)
    )


def paired(champion_lineage=None, rows=None):
    challenger = SpyModel()
    champion = SpyModel()
    split = split_rows(synthetic_rows(), config())
    previous = split_rows(synthetic_rows(2023, 100), config(2023))
    result = compare_models(
        challenger,
        feature_schema(2025),
        split.metadata,
        champion,
        feature_schema(2024),
        champion_lineage if champion_lineage is not None else previous.metadata,
        split.test if rows is None else rows,
    )
    return result, challenger, champion


def test_comparison_identical_rows_order_labels_and_own_schemas():
    result, challenger, champion = paired()
    assert result["status"] == "verified"
    left, right = challenger.frames[0], champion.frames[0]
    pd.testing.assert_frame_equal(
        left.drop(columns="building_age"), right.drop(columns="building_age")
    )
    assert (left["building_age"] - right["building_age"]).tolist() == [1, 1, 1]
    assert result["challenger_metrics"] == result["champion_metrics"]
    assert result["evaluation"]["count"] == 3


@pytest.mark.parametrize("lineage", [{}, {"version": 1}, {"version": 2}])
def test_unknown_lineage_never_calls_predict(lineage):
    result, challenger, champion = paired(lineage)
    assert result["status"] == "unverifiable"
    assert not challenger.frames and not champion.frames


def test_fit_and_validation_property_exposure_and_temporal_overlap():
    for index in (0, 3):
        rows = synthetic_rows(2023, 100)
        rows[index]["id"] = 6
        lineage = split_rows(rows, config(2023)).metadata
        result, new, old = paired(lineage)
        assert result["reason"] == "champion_property_overlap"
        assert not new.frames and not old.frames
    lineage = split_rows(synthetic_rows(2025, 100), config()).metadata
    # Same holdout is admissible: only fit/validation exposure matters.
    assert paired(lineage)[0]["status"] == "verified"
    future = split_rows(synthetic_rows(2026, 100), config(2026)).metadata
    assert paired(future)[0]["reason"] == "holdout_not_after_champion_exposure"


def test_partial_holdout_and_changed_labels_never_silently_intersect():
    rows = split_rows(synthetic_rows(), config()).test
    for changed in (rows[:1], [dict(rows[0], price=999)] + rows[1:]):
        result, new, old = paired(rows=changed)
        assert result["reason"] == "evaluation_membership_mismatch"
        assert not new.frames and not old.frames


def test_bad_prediction_count_is_rejected():
    split = split_rows(synthetic_rows(), config())
    model = Mock()
    model.predict.return_value = [0]
    with pytest.raises(ValueError, match="one finite"):
        compare_models(
            model,
            feature_schema(2025),
            split.metadata,
            SpyModel(),
            feature_schema(2023),
            split_rows(synthetic_rows(2023, 100), config(2023)).metadata,
            split.test,
        )


def test_champion_loader_does_not_treat_unknown_as_absent():
    storage = Mock()
    storage.get_json.side_effect = RuntimeError("synthetic failure")
    result = compare_with_champion(storage, SpyModel(), feature_schema(2025), {}, [])
    assert result == {"status": "unverifiable", "reason": "champion_unavailable_or_invalid"}
    storage.get_bytes.assert_not_called()


def test_all_constant_features_rejected_before_model_creation(monkeypatch):
    rows = synthetic_rows()
    for i in (1, 2):
        rows[i] = dict(
            rows[0],
            id=rows[i]["id"],
            turnover_date=rows[i]["turnover_date"],
            price=rows[i]["price"],
        )
    factory = Mock()
    monkeypatch.setattr("app.training.modeling.CatBoostRegressor", factory)
    with pytest.raises(ValueError, match="varying train feature"):
        train_and_evaluate(rows, reference_year=2025, split_config=config())
    factory.assert_not_called()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda x: x["partitions"]["validation"].update(membership=[]),
        lambda x: x["partitions"]["train"].update(count=99),
        lambda x: x["partitions"]["validation"]["membership"][0].update(id=999),
        lambda x: x["partitions"]["train"]["period"].update(end="2030-01-01"),
        lambda x: x.update(training_mode="warm_start"),
        lambda x: x.pop("boundaries"),
        lambda x: x.pop("dataset"),
    ],
)
def test_corrupt_or_partial_exposure_never_compares(mutation):
    lineage = split_rows(synthetic_rows(2023, 100), config(2023)).metadata
    mutation(lineage)
    result, new, old = paired(lineage)
    assert result["status"] == "unverifiable"
    assert not new.frames and not old.frames


def test_missing_settings_rejected_before_storage_initialization(monkeypatch):
    from app.config import settings
    from app.training.pipeline import Pipeline

    monkeypatch.setattr(settings, "split_train_start", None)
    storage = Mock()
    monkeypatch.setattr("app.training.pipeline.S3Storage", storage)
    with pytest.raises(ValueError, match="Require"):
        Pipeline()
    storage.assert_not_called()


def champion_bundle():
    previous = split_rows(synthetic_rows(2023, 100), config(2023))
    pointer = {
        "model_version": "A",
        "type": "sklearn",
        "artifact_key": "models/A/model.pkl",
        "snapshot_prefix": "synthetic-champion",
        "created_at": "2024-01-01T00:00:00+00:00",
    }
    manifest = {
        "model_version": "A",
        "snapshot_prefix": "synthetic-champion",
        "period": {"start_date": "2023-01-01", "end_date": "2023-06-30"},
        "counts": {"rows_trainable": 8},
        "dropped_reasons": {},
        "metrics": {},
        "gating": {"passed": True, "reasons": [], "details": {}},
        "lineage": previous.metadata,
    }
    return pointer, manifest, feature_schema(2024)


@pytest.mark.parametrize(
    "target,field,value",
    [
        ("pointer", "artifact_key", "models/B/model.pkl"),
        ("manifest", "model_version", "B"),
        ("pointer", "model_version", "<missing>"),
        ("pointer", "artifact_key", "<missing>"),
        ("manifest", "model_version", "<missing>"),
        *[
            ("pointer", "model_version", value)
            for value in (
                None,
                "",
                " ",
                True,
                123,
                [],
                {},
                ".",
                "..",
                "A/B",
                " A",
                "A ",
                "A\n",
                "Å",
            )
        ],
        *[
            ("pointer", "artifact_key", value)
            for value in (
                None,
                "",
                True,
                "models/A/../B/model.pkl",
                "models/A/model.pkl/",
                "models/A/other.pkl",
            )
        ],
        *[("manifest", "model_version", value) for value in (None, "", True, 123)],
    ],
)
def test_champion_identity_rejected_before_deserialization(monkeypatch, target, field, value):
    from app.config import settings
    from app.training.gating import evaluate_publish_gate

    pointer, manifest, schema = champion_bundle()
    obj = pointer if target == "pointer" else manifest
    if value == "<missing>":
        obj.pop(field)  # Missing fields must not be inferred from another reference.
    else:
        obj[field] = value
    storage = Mock()
    storage.get_json.side_effect = [pointer, manifest, schema]
    loader = Mock(return_value=SpyModel())
    predict = Mock(side_effect=AssertionError("Prediction forbidden for invalid identity"))
    monkeypatch.setattr("app.training.comparison.joblib.load", loader)
    monkeypatch.setattr(SpyModel, "predict", predict)
    split = split_rows(synthetic_rows(), config())
    result = compare_with_champion(
        storage, SpyModel(), feature_schema(2025), split.metadata, split.test
    )
    assert result == {"status": "unverifiable", "reason": "champion_unavailable_or_invalid"}
    storage.get_bytes.assert_not_called()
    loader.assert_not_called()
    predict.assert_not_called()
    monkeypatch.setattr(settings, "train_min_rows", 0)
    gate = evaluate_publish_gate(
        rows_trainable=8, new_metrics={}, prev_metrics=None, comparison=result
    )
    assert not gate.passed
    assert gate.reasons == ["unverifiable_comparison"]


def test_champion_same_version_bundle_deserializes_and_evaluates(monkeypatch):
    from app.config import settings
    from app.training.gating import evaluate_publish_gate

    pointer, manifest, schema = champion_bundle()
    storage = Mock()
    storage.get_json.side_effect = [pointer, manifest, schema]
    storage.get_bytes.return_value = b"synthetic-model-bytes"
    champion = SpyModel()
    challenger = SpyModel()
    loader = Mock(return_value=champion)
    monkeypatch.setattr("app.training.comparison.joblib.load", loader)
    split = split_rows(synthetic_rows(), config())
    result = compare_with_champion(
        storage, challenger, feature_schema(2025), split.metadata, split.test
    )
    assert result["status"] == "verified"
    assert result["champion_reference"] == pointer
    loader.assert_called_once()
    storage.get_bytes.assert_called_once_with(
        bucket=settings.s3_bucket_models, key="models/A/model.pkl"
    )
    assert len(champion.frames) == len(challenger.frames) == 1
    monkeypatch.setattr(settings, "train_min_rows", 0)
    assert evaluate_publish_gate(
        rows_trainable=8, new_metrics={}, prev_metrics=None, comparison=result
    ).passed
