"""Synthetic policy and evidence; no fitted model or external storage."""

import json
from copy import deepcopy
from unittest.mock import Mock

import pytest

from app.features import feature_schema
from app.storage.s3 import S3ObjectNotFound, S3StorageError
from app.training.comparison import compare_with_champion
from app.training.gating import evaluate_publish_gate
from app.training.splitting import split_rows
from tests.test_temporal_evaluation import config, synthetic_rows

POLICY = {
    "version": 1,
    "bootstrap": False,
    "min_trainable": 8,
    "min_evaluation": 3,
    "segments": {"enebolig": 1},
    "metrics": {name: {"max": 10, "max_degradation": 0.1} for name in ("mdape", "ae_p90", "wape")},
}


def evidence():
    lineage = split_rows(synthetic_rows(), config()).metadata
    evaluation = lineage["partitions"]["test"]
    group = dict.fromkeys(POLICY["metrics"], 1.0)
    metrics = {
        "overall": deepcopy(group),
        "by_realestate_type": {"enebolig": deepcopy(group)},
        "evaluation": evaluation,
        "n_test": evaluation["count"],
    }
    paired = {k: deepcopy(metrics[k]) for k in ("overall", "by_realestate_type")}
    comparison = {
        "status": "verified",
        "evaluation": evaluation,
        "evaluation_order": [m["row_hash"] for m in evaluation["membership"]],
        "feature_schemas": {"champion": feature_schema(2024), "challenger": feature_schema(2025)},
        "champion_lineage_fingerprint": "a" * 64,
        "champion_reference": {
            "model_version": "A",
            "type": "sklearn",
            "artifact_key": "models/A/model.pkl",
        },
        "champion_metrics": deepcopy(paired),
        "challenger_metrics": paired,
    }
    return lineage, metrics, comparison


def decide(policy=POLICY, metrics=None, comparison=None, lineage=None, rows=8):
    lin, met, comp = evidence()
    return evaluate_publish_gate(
        rows_trainable=rows,
        new_metrics=met if metrics is None else metrics,
        comparison=comp if comparison is None else comparison,
        lineage=lin if lineage is None else lineage,
        policy=json.dumps(policy) if policy is not None else "",
    )


def test_valid_evidence_passes_and_serializes():
    result = decide()
    assert result.passed
    json.dumps(result.__dict__, allow_nan=False)


@pytest.mark.parametrize(
    "policy",
    [
        None,
        {},
        {**POLICY, "bootstrap": 1},
        {**POLICY, "min_evaluation": True},
        {**POLICY, "metrics": {}},
        {**POLICY, "segments": {}},
        {**POLICY, "min_trainable": -1},
    ],
)
def test_bad_policy_blocks(policy):
    assert decide(policy=policy).reasons == ["policy_unconfigured_or_invalid"]


@pytest.mark.parametrize("value", [None, True, "1", float("nan"), float("inf"), -1])
def test_bad_metric_blocks(value):
    _, metrics, comparison = evidence()
    metrics["overall"]["mdape"] = value
    comparison["challenger_metrics"]["overall"]["mdape"] = value
    result = decide(metrics=metrics, comparison=comparison)
    assert not result.passed
    json.dumps(result.__dict__, allow_nan=False)


def test_boundaries_and_zero():
    _, metrics, comparison = evidence()
    metrics["overall"]["mdape"] = 1.1
    comparison["challenger_metrics"]["overall"]["mdape"] = 1.1
    assert decide(metrics=metrics, comparison=comparison).passed
    metrics["overall"]["mdape"] = 1.11
    comparison["challenger_metrics"]["overall"]["mdape"] = 1.11
    assert "relative_limit_exceeded" in decide(metrics=metrics, comparison=comparison).reasons
    comparison["champion_metrics"]["overall"]["mdape"] = 0
    result = decide(metrics=metrics, comparison=comparison)
    assert "relative_limit_exceeded" in result.reasons
    json.dumps(result.__dict__, allow_nan=False)
    metrics["overall"]["mdape"] = 0
    comparison["challenger_metrics"]["overall"]["mdape"] = 0
    assert decide(metrics=metrics, comparison=comparison).passed


def test_counts_and_missing_segment():
    assert "trainable_count_mismatch" in decide(rows=7).reasons
    assert "insufficient_trainable_count" in decide(policy={**POLICY, "min_trainable": 9}).reasons
    assert (
        "insufficient_evaluation_count" in decide(policy={**POLICY, "min_evaluation": 4}).reasons
    )
    assert (
        "insufficient_segment_count"
        in decide(policy={**POLICY, "segments": {"missing": 1}}).reasons
    )


def test_comparison_evidence_must_match():
    _, metrics, comp = evidence()
    comp["evaluation"] = {"fingerprint": "other"}
    assert "invalid_comparison_evidence" in decide(metrics=metrics, comparison=comp).reasons
    comp = {"status": "verified"}
    assert "invalid_comparison_evidence" in decide(comparison=comp).reasons


def test_bootstrap_only_explicit_missing_pointer():
    missing = {"status": "no_active_pointer"}
    assert "bootstrap_disabled" in decide(comparison=missing).reasons
    assert decide(policy={**POLICY, "bootstrap": True}, comparison=missing).passed
    assert not decide(
        policy={**POLICY, "bootstrap": True}, comparison={"status": "unverifiable"}
    ).passed


@pytest.mark.parametrize(
    "error,expected",
    [
        (S3ObjectNotFound(), "no_active_pointer"),
        (S3StorageError(), "unverifiable"),
        (TimeoutError(), "unverifiable"),
    ],
)
def test_pointer_errors(error, expected):
    storage = Mock()
    storage.get_json.side_effect = error
    assert compare_with_champion(storage, None, None, None, None)["status"] == expected


@pytest.mark.parametrize(
    "code,expected",
    [
        ("NoSuchKey", "no_active_pointer"),
        ("NoSuchBucket", "unverifiable"),
        ("AccessDenied", "unverifiable"),
        ("403", "unverifiable"),
        ("404", "unverifiable"),
    ],
)
def test_s3_get_error_classification(code, expected):
    from botocore.exceptions import ClientError

    from app.storage.s3 import S3Storage

    storage = S3Storage.__new__(S3Storage)
    storage._client = Mock()
    storage._client.get_object.side_effect = ClientError({"Error": {"Code": code}}, "GetObject")
    assert compare_with_champion(storage, None, None, None, None)["status"] == expected


def test_candidate_identity_count_and_absolute_limit():
    _, metrics, comp = evidence()
    metrics["n_test"] = True
    assert "invalid_evaluation_evidence" in decide(metrics=metrics, comparison=comp).reasons
    _, metrics, comp = evidence()
    metrics["overall"]["mdape"] = 11
    comp["challenger_metrics"]["overall"]["mdape"] = 11
    assert "absolute_limit_exceeded" in decide(metrics=metrics, comparison=comp).reasons


@pytest.mark.parametrize("rows", [7, 9, 500])
def test_trainable_count_must_match_lineage(rows):
    policy = {**POLICY, "min_trainable": 500 if rows == 500 else 8}
    result = decide(rows=rows, policy=policy)
    assert not result.passed
    assert "trainable_count_mismatch" in result.reasons
    assert result.details["lineage_trainable_count"] == 8
    json.dumps(result.__dict__, allow_nan=False)


def test_consistent_trainable_count_at_boundary_passes():
    assert decide(rows=8, policy={**POLICY, "min_trainable": 8}).passed


def test_bootstrap_rejects_trainable_count_mismatch():
    result = decide(
        rows=500,
        policy={**POLICY, "min_trainable": 500, "bootstrap": True},
        comparison={"status": "no_active_pointer"},
    )
    assert not result.passed
    assert "trainable_count_mismatch" in result.reasons
    json.dumps(result.__dict__, allow_nan=False)


@pytest.mark.parametrize("bad", [None, {}, [], "bad", "g" * 64, "a" * 63])
def test_malformed_evaluation_order_rejected_safely(bad):
    _, metrics, comparison = evidence()
    comparison["evaluation_order"][0] = bad
    result = decide(metrics=metrics, comparison=comparison)
    assert not result.passed
    assert "invalid_comparison_evidence" in result.reasons
    assert "evaluation_order" not in result.details
    json.dumps(result.__dict__, allow_nan=False)


def test_valid_evaluation_order_passes():
    _, metrics, comparison = evidence()
    assert decide(metrics=metrics, comparison=comparison).passed
