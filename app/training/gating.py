"""Fail-closed promotion decisions over explicitly configured evidence."""

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

from app.config import settings
from app.features import validate_feature_schema
from app.training.comparison import validate_lineage


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reasons: list[str]
    details: dict[str, Any]


def _number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def _policy(raw):
    if not isinstance(raw, str) or not raw:
        raise ValueError
    obj = json.loads(raw)
    if not isinstance(obj, dict) or set(obj) != {
        "version",
        "bootstrap",
        "min_trainable",
        "min_evaluation",
        "segments",
        "metrics",
    }:
        raise ValueError
    if (
        type(obj["version"]) is not int
        or obj["version"] != 1
        or type(obj["bootstrap"]) is not bool
    ):
        raise ValueError
    for key in ("min_trainable", "min_evaluation"):
        if type(obj[key]) is not int or obj[key] < 1:
            raise ValueError
    if not isinstance(obj["segments"], dict) or not obj["segments"]:
        raise ValueError
    if any(
        not isinstance(k, str) or not k or type(v) is not int or v < 1
        for k, v in obj["segments"].items()
    ):
        raise ValueError
    metrics = obj["metrics"]
    if not isinstance(metrics, dict) or set(metrics) != {"mdape", "ae_p90", "wape"}:
        raise ValueError
    for rule in metrics.values():
        if not isinstance(rule, dict) or set(rule) != {"max", "max_degradation"}:
            raise ValueError
        if not all(_number(rule[k]) for k in rule) or rule["max_degradation"] > 1:
            raise ValueError
    return obj


def evaluate_publish_gate(
    *, rows_trainable, new_metrics, prev_metrics=None, comparison=None, lineage=None, policy=None
):
    reasons = []
    details = {"checks": {}}
    try:
        p = _policy(settings.promotion_policy if policy is None else policy)
    except (ValueError, TypeError, json.JSONDecodeError, OverflowError):
        return GateResult(False, ["policy_unconfigured_or_invalid"], details)
    details["policy"] = p
    details["policy_id"] = hashlib.sha256(
        json.dumps(p, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    if type(rows_trainable) is int and 0 <= rows_trainable <= 2**63 - 1:
        details["trainable_count"] = rows_trainable
    valid_trainable_count = type(rows_trainable) is int and 0 <= rows_trainable <= 2**63 - 1
    if not valid_trainable_count:
        reasons.append("insufficient_trainable_count")
    try:
        validate_lineage(lineage)
        partitions = lineage["partitions"]
        dataset_count = sum(partitions[name]["count"] for name in ("train", "validation", "test"))
        details["lineage_trainable_count"] = dataset_count
        if valid_trainable_count and rows_trainable != dataset_count:
            reasons.append("trainable_count_mismatch")
        elif valid_trainable_count and dataset_count < p["min_trainable"]:
            reasons.append("insufficient_trainable_count")
        evaluation = partitions["test"]
        if (
            not isinstance(new_metrics, dict)
            or new_metrics.get("evaluation") != evaluation
            or type(new_metrics.get("n_test")) is not int
            or new_metrics.get("n_test") != evaluation["count"]
        ):
            raise ValueError
        details["evaluation_count"] = evaluation["count"]
        details["segment_counts"] = {
            name: evaluation["segments"].get(name, 0) for name in p["segments"]
        }
        if evaluation["count"] < p["min_evaluation"]:
            reasons.append("insufficient_evaluation_count")
        for segment, minimum in p["segments"].items():
            if evaluation["segments"].get(segment, 0) < minimum:
                reasons.append("insufficient_segment_count")
        details["evaluation_id"] = evaluation["fingerprint"]
    except (ValueError, KeyError, TypeError, AttributeError):
        return GateResult(
            False, list(dict.fromkeys(reasons + ["invalid_evaluation_evidence"])), details
        )

    bootstrap = isinstance(comparison, dict) and comparison.get("status") == "no_active_pointer"
    if bootstrap:
        if not p["bootstrap"]:
            reasons.append("bootstrap_disabled")
    elif not isinstance(comparison, dict) or comparison.get("status") != "verified":
        reasons.append("unverifiable_comparison")
    else:
        order = comparison.get("evaluation_order")
        if (
            comparison.get("evaluation") != evaluation
            or not isinstance(order, list)
            or len(order) != evaluation["count"]
            or any(
                not isinstance(x, str)
                or len(x) != 64
                or any(c not in "0123456789abcdef" for c in x)
                for x in order
            )
            or sorted(order) != sorted(m["row_hash"] for m in evaluation["membership"])
            or not isinstance(comparison.get("feature_schemas"), dict)
            or not isinstance(comparison.get("champion_lineage_fingerprint"), str)
            or not isinstance(comparison.get("champion_reference"), dict)
            or not isinstance(comparison.get("champion_metrics"), dict)
            or not isinstance(comparison.get("challenger_metrics"), dict)
            or any(
                comparison["challenger_metrics"].get(k) != new_metrics.get(k)
                for k in ("overall", "by_realestate_type")
            )
        ):
            reasons.append("invalid_comparison_evidence")
        else:
            try:
                schemas = comparison["feature_schemas"]
                validate_feature_schema(schemas["champion"])
                validate_feature_schema(schemas["challenger"])
                fp = comparison["champion_lineage_fingerprint"]
                if len(fp) != 64 or any(c not in "0123456789abcdef" for c in fp):
                    raise ValueError
                ref = comparison["champion_reference"]
                version = ref["model_version"]
                import re

                if (
                    not isinstance(version, str)
                    or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", version)
                    or ref.get("type") != "sklearn"
                    or ref.get("artifact_key") != f"models/{version}/model.pkl"
                ):
                    raise ValueError
            except (ValueError, KeyError, TypeError, AttributeError):
                reasons.append("invalid_comparison_evidence")
            prev_metrics = comparison["champion_metrics"]
    for segment in [None, *p["segments"]]:
        for metric, rule in p["metrics"].items():
            name = f"{segment or 'overall'}.{metric}"

            def value(metrics, segment=segment, metric=metric):
                if not isinstance(metrics, dict):
                    return None
                group = (
                    metrics.get("overall")
                    if segment is None
                    else metrics.get("by_realestate_type", {}).get(segment)
                    if isinstance(metrics.get("by_realestate_type"), dict)
                    else None
                )
                return group.get(metric) if isinstance(group, dict) else None

            new = value(new_metrics)
            old = value(prev_metrics)
            if not _number(new):
                reasons.append("invalid_candidate_metric")
                continue
            check = {"candidate": new, "absolute_max": rule["max"]}
            details["checks"][name] = check
            if new > rule["max"]:
                reasons.append("absolute_limit_exceeded")
            if (
                not bootstrap
                and "invalid_comparison_evidence" not in reasons
                and "unverifiable_comparison" not in reasons
            ):
                if not _number(old):
                    reasons.append("invalid_champion_metric")
                else:
                    check["champion"] = old
                    check["max_degradation"] = rule["max_degradation"]
                    if old == 0:
                        check["zero_champion"] = True
                        if new > 0:
                            reasons.append("relative_limit_exceeded")
                    elif new > old * (1 + rule["max_degradation"]):
                        reasons.append("relative_limit_exceeded")
    return GateResult(not reasons, list(dict.fromkeys(reasons)), details)
