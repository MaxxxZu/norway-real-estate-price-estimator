"""Paired evaluation against a trusted, explicitly described fresh-fit champion."""

import re
from datetime import date
from io import BytesIO

import joblib

from app.config import settings
from app.features import validate_feature_schema
from app.training.modeling import evaluate_model
from app.training.snapshots import semantic_fingerprint
from app.training.splitting import TemporalSplitConfig, membership


def _is_hash(value):
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def validate_lineage(lineage):
    """Reject partial/legacy exposure records rather than interpreting them as empty."""
    if lineage["version"] != 1 or lineage["training_mode"] != "fresh_fit":
        raise ValueError("Unsupported training lineage")
    if lineage["property_overlap_policy"] != "reject_cross_partition":
        raise ValueError("Unsupported overlap policy")
    config = TemporalSplitConfig(
        **{key: date.fromisoformat(value) for key, value in lineage["boundaries"].items()}
    )
    boundaries = list(config.as_dict().values())
    ids = []
    for name, start, end in zip(("train", "validation", "test"), boundaries, boundaries[1:]):
        part = lineage["partitions"][name]
        members = part["membership"]
        if (
            not isinstance(members, list)
            or not members
            or type(part["count"]) is not int
            or part["count"] != len(members)
        ):
            raise ValueError("Incomplete exposure membership")
        if (
            not _is_hash(part["fingerprint"])
            or not isinstance(part["segments"], dict)
            or any(type(count) is not int or count <= 0 for count in part["segments"].values())
            or sum(part["segments"].values()) != part["count"]
        ):
            raise ValueError("Incomplete exposure membership")
        if any(
            type(m["id"]) is not int
            or m["id"] <= 0
            or not start <= date.fromisoformat(m["date"]).isoformat() < end
            or not _is_hash(m["row_hash"])
            for m in members
        ):
            raise ValueError("Invalid exposure membership")
        if part["membership_fingerprint"] != semantic_fingerprint(members):
            raise ValueError("Exposure membership fingerprint mismatch")
        if part["period"] != {
            "start": min(m["date"] for m in members),
            "end": max(m["date"] for m in members),
        }:
            raise ValueError("Exposure period mismatch")
        ids.append({m["id"] for m in members})
    if any(ids[a] & ids[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Invalid cross-partition exposure")
    if not _is_hash(lineage["dataset"]["fingerprint"]):
        raise ValueError("Missing dataset identity")
    if lineage["overlaps"] != {"train:validation": [], "train:test": [], "validation:test": []}:
        raise ValueError("Invalid recorded overlap checks")


def compare_models(
    challenger,
    challenger_schema,
    challenger_lineage,
    champion,
    champion_schema,
    champion_lineage,
    rows,
):
    try:
        validate_lineage(champion_lineage)
        validate_lineage(challenger_lineage)
        validate_feature_schema(champion_schema)
        validate_feature_schema(challenger_schema)
    except (ValueError, KeyError, TypeError, AttributeError):
        return {"status": "unverifiable", "reason": "unknown_or_invalid_lineage_or_schema"}
    evaluation = challenger_lineage["partitions"]["test"]
    if (
        membership(rows) != evaluation["membership"]
        or semantic_fingerprint(rows) != evaluation["fingerprint"]
    ):
        return {"status": "unverifiable", "reason": "evaluation_membership_mismatch"}
    exposed = [
        m
        for name in ("train", "validation")
        for m in champion_lineage["partitions"][name]["membership"]
    ]
    if {m["id"] for m in exposed} & {row["id"] for row in rows}:
        return {"status": "unverifiable", "reason": "champion_property_overlap"}
    if min(row["turnover_date"] for row in rows) <= max(m["date"] for m in exposed):
        return {"status": "unverifiable", "reason": "holdout_not_after_champion_exposure"}
    # Neither transform is allowed to select/filter rows. Prediction shape is checked.
    new_metrics = evaluate_model(challenger, rows, challenger_schema)
    prev_metrics = evaluate_model(champion, rows, champion_schema)
    return {
        "status": "verified",
        "evaluation": evaluation,
        "evaluation_order": [semantic_fingerprint([row]) for row in rows],
        "feature_schemas": {"champion": champion_schema, "challenger": challenger_schema},
        "champion_lineage_fingerprint": semantic_fingerprint([champion_lineage]),
        "challenger_metrics": new_metrics,
        "champion_metrics": prev_metrics,
    }


def compare_with_champion(storage, challenger, schema, lineage, rows):
    """Read a single champion reference. Missing/unreadable is never a bootstrap decision."""
    try:
        latest = storage.get_json(bucket=settings.s3_bucket_models, key="latest.json")
        if not isinstance(latest, dict):
            raise ValueError("Invalid champion pointer")
        version = latest.get("model_version")
        if (
            not isinstance(version, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", version) is None
            or latest.get("type") != "sklearn"
        ):
            raise ValueError("Invalid champion identity")
        prefix = f"models/{version}"
        model_key = f"{prefix}/model.pkl"
        if latest.get("artifact_key") != model_key:
            raise ValueError("Champion artifact identity mismatch")
        manifest = storage.get_json(
            bucket=settings.s3_bucket_models, key=f"{prefix}/training_manifest.json"
        )
        if not isinstance(manifest, dict) or manifest.get("model_version") != version:
            raise ValueError("Champion manifest identity mismatch")
        champion_lineage = manifest["lineage"]
        champion_schema = storage.get_json(
            bucket=settings.s3_bucket_models, key=f"{prefix}/feature_schema.json"
        )
        validate_lineage(champion_lineage)
        validate_feature_schema(champion_schema)
        champion = joblib.load(
            BytesIO(storage.get_bytes(bucket=settings.s3_bucket_models, key=model_key))
        )
        comparison = compare_models(
            challenger, schema, lineage, champion, champion_schema, champion_lineage, rows
        )
        comparison["champion_reference"] = latest
        return comparison
    except Exception:
        # No exception text: SDK/deserialization errors can contain sensitive details.
        return {"status": "unverifiable", "reason": "champion_unavailable_or_invalid"}
