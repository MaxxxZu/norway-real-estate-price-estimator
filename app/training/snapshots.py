"""Immutable snapshot revisions. Completion manifests are always written last."""

import hashlib
import json
import math
from dataclasses import dataclass
from io import BytesIO
from typing import Any
from uuid import uuid4

import pandas as pd

from app.config import settings
from app.storage.s3 import S3Storage, S3StorageError
from app.training.dataset import NORMALIZATION_VERSION, build_trainable_dataset

SNAPSHOT_VERSION = 1


@dataclass(frozen=True)
class SnapshotPaths:
    prefix: str
    raw_rows_key: str
    dataset_key: str
    manifest_key: str


@dataclass(frozen=True)
class RawSnapshotPaths:
    prefix: str
    raw_rows_key: str
    manifest_key: str
    start_date: str
    end_date: str


def _canonical_value(value: Any) -> Any:
    if isinstance(value, dict):
        # Wrap actual objects so a user object cannot impersonate a float tag.
        return {"__object__": {key: _canonical_value(val) for key, val in value.items()}}
    if isinstance(value, list):
        return [_canonical_value(val) for val in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return {"__nonfinite_float__": str(value)}
        if value.is_integer():
            return int(value)
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        _canonical_value(value),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=canonical_json)


def semantic_fingerprint(rows: list[dict[str, Any]]) -> str:
    """Order-independent multiset hash; duplicates are significant."""
    return hashlib.sha256(canonical_json(canonical_rows(rows)).encode()).hexdigest()


def snapshot_paths_for_prefix(prefix: str) -> SnapshotPaths:
    return SnapshotPaths(
        prefix, f"{prefix}/rows_raw.jsonl", f"{prefix}/dataset.parquet", f"{prefix}/manifest.json"
    )


def snapshot_paths(start_date: str, end_date: str) -> SnapshotPaths:
    return snapshot_paths_for_prefix(
        f"snapshots/v1/{start_date}_{end_date}/revisions/{uuid4().hex}"
    )


def raw_snapshot_paths(start_date: str, end_date: str, revision: str) -> RawSnapshotPaths:
    prefix = f"snapshots/v1/{start_date}_{end_date}/revisions/{revision}"
    return RawSnapshotPaths(
        prefix, f"{prefix}/rows_raw.jsonl", f"{prefix}/manifest.json", start_date, end_date
    )


def raw_latest_key(start_date: str, end_date: str) -> str:
    return f"snapshots/v1/{start_date}_{end_date}/latest_raw.json"


def resolve_raw_snapshot(
    storage: S3Storage, start_date: str, end_date: str
) -> RawSnapshotPaths | None:
    key = raw_latest_key(start_date, end_date)
    if not storage.exists(settings.s3_bucket_snapshots, key):
        return None
    ref = storage.get_json(bucket=settings.s3_bucket_snapshots, key=key)
    paths = raw_snapshot_paths(start_date, end_date, ref["revision"])
    return paths if raw_snapshot_exists(storage, paths) else None


def upload_jsonl(storage: S3Storage, bucket: str, key: str, rows: list[dict[str, Any]]) -> None:
    # Raw data may include rejected NaN/Infinity; retained for diagnostics/re-normalization.
    data = "\n".join(json.dumps(r, ensure_ascii=False) for r in canonical_rows(rows)) + "\n"
    storage.put_bytes(
        bucket=bucket, key=key, data=data.encode(), content_type="application/x-ndjson"
    )


def upload_parquet(storage: S3Storage, bucket: str, key: str, rows: list[dict[str, Any]]) -> None:
    data = pd.DataFrame(canonical_rows(rows)).to_parquet(index=False)
    storage.put_bytes(bucket=bucket, key=key, data=data, content_type="application/octet-stream")


def upload_manifest(storage: S3Storage, bucket: str, key: str, manifest: dict[str, Any]) -> None:
    storage.put_json(bucket=bucket, key=key, obj=manifest)


def iter_jsonl_rows(storage: S3Storage, key: str):
    for line in storage.iter_lines(bucket=settings.s3_bucket_snapshots, key=key):
        if line.strip():
            yield json.loads(line)


def load_jsonl_rows(storage: S3Storage, key: str) -> list[dict[str, Any]]:
    return list(iter_jsonl_rows(storage, key))


def upload_snapshots(
    storage: S3Storage,
    start_date: str,
    end_date: str,
    raw_rows: list[dict[str, Any]],
    trainable_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> SnapshotPaths:
    paths = snapshot_paths(start_date, end_date)
    return upload_snapshots_with_prefix(storage, paths.prefix, raw_rows, trainable_rows, manifest)


def upload_snapshots_with_prefix(
    storage: S3Storage,
    prefix: str,
    raw_rows: list[dict[str, Any]],
    trainable_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> SnapshotPaths:
    paths = snapshot_paths_for_prefix(prefix)
    if storage.exists(settings.s3_bucket_snapshots, paths.manifest_key):
        existing = load_manifest(storage, paths.manifest_key)
        if (
            existing["fingerprint"] != semantic_fingerprint(trainable_rows)
            or existing["raw_fingerprint"] != semantic_fingerprint(raw_rows)
            or existing.get("source_months", []) != manifest.get("source_months", [])
        ):
            raise RuntimeError("Immutable snapshot conflict")
        if not snapshot_exists(storage, paths):
            raise RuntimeError("Completed snapshot has missing files")
        return paths
    upload_jsonl(storage, settings.s3_bucket_snapshots, paths.raw_rows_key, raw_rows)
    upload_parquet(storage, settings.s3_bucket_snapshots, paths.dataset_key, trainable_rows)
    completed = {
        **manifest,
        "snapshot_version": SNAPSHOT_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "complete": True,
        "revision": prefix.rsplit("/", 1)[-1],
        "fingerprint": semantic_fingerprint(trainable_rows),
        "raw_fingerprint": semantic_fingerprint(raw_rows),
    }
    upload_manifest(storage, settings.s3_bucket_snapshots, paths.manifest_key, completed)
    return paths


def upload_raw_snapshot(
    storage: S3Storage,
    start_date: str,
    end_date: str,
    raw_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> RawSnapshotPaths:
    revision = uuid4().hex
    paths = raw_snapshot_paths(start_date, end_date, revision)
    upload_jsonl(storage, settings.s3_bucket_snapshots, paths.raw_rows_key, raw_rows)
    completed = {
        **manifest,
        "period": {"start_date": start_date, "end_date": end_date},
        "snapshot_version": SNAPSHOT_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "complete": True,
        "revision": revision,
        "raw_fingerprint": semantic_fingerprint(raw_rows),
    }
    upload_manifest(storage, settings.s3_bucket_snapshots, paths.manifest_key, completed)
    storage.put_json(
        bucket=settings.s3_bucket_snapshots,
        key=raw_latest_key(start_date, end_date),
        obj={"revision": revision},
    )
    return paths


def snapshot_exists(storage: S3Storage, paths: SnapshotPaths) -> bool:
    return _complete(storage, paths.manifest_key, dataset=True) and all(
        storage.exists(settings.s3_bucket_snapshots, key)
        for key in (paths.raw_rows_key, paths.dataset_key)
    )


def raw_snapshot_exists(storage: S3Storage, paths: RawSnapshotPaths) -> bool:
    return _complete(storage, paths.manifest_key) and storage.exists(
        settings.s3_bucket_snapshots, paths.raw_rows_key
    )


def _valid_manifest(manifest: Any, key: str, *, dataset: bool = False) -> bool:
    if not isinstance(manifest, dict):
        return False
    for field, expected in (
        ("snapshot_version", SNAPSHOT_VERSION),
        ("normalization_version", NORMALIZATION_VERSION),
    ):
        if type(manifest.get(field)) is not int or manifest[field] != expected:
            return False
    for field in ["raw_fingerprint", "fingerprint"] if dataset else ["raw_fingerprint"]:
        value = manifest.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
        ):
            return False
    return manifest.get("complete") is True and manifest.get("revision") == key.rsplit("/", 2)[-2]


def _complete(storage: S3Storage, key: str, *, dataset: bool = False) -> bool:
    if not storage.exists(settings.s3_bucket_snapshots, key):
        return False
    manifest = storage.get_json(bucket=settings.s3_bucket_snapshots, key=key)
    return _valid_manifest(manifest, key, dataset=dataset)


def load_manifest(storage: S3Storage, manifest_key: str) -> dict[str, Any]:
    try:
        manifest = storage.get_json(bucket=settings.s3_bucket_snapshots, key=manifest_key)
    except S3StorageError as exc:
        raise RuntimeError("Snapshot manifest unavailable; fetch a new revision") from exc
    if not _valid_manifest(manifest, manifest_key):
        raise RuntimeError("Legacy, incompatible or incomplete snapshot; fetch a new revision")
    return manifest


def load_trainable_rows_from_parquet(storage: S3Storage, dataset_key: str) -> list[dict[str, Any]]:
    paths = snapshot_paths_for_prefix(dataset_key.rsplit("/", 1)[0])
    manifest = load_manifest(storage, paths.manifest_key)
    if not snapshot_exists(storage, paths):
        raise RuntimeError("Incomplete snapshot")
    raw = storage.get_bytes(bucket=settings.s3_bucket_snapshots, key=dataset_key)
    df = pd.read_parquet(BytesIO(raw))
    # Parquet nullable integers can return NaN rather than None.
    rows = df.astype(object).where(pd.notna(df), None).to_dict(orient="records")
    result = build_trainable_dataset(rows)
    if (
        result.dropped_reasons
        or semantic_fingerprint(result.trainable_rows) != manifest["fingerprint"]
    ):
        raise RuntimeError("Snapshot normalized data/fingerprint mismatch")
    return canonical_rows(result.trainable_rows)


def fetch_latest_snapshot_ref(storage: S3Storage) -> SnapshotPaths:
    latest = storage.get_json(bucket=settings.s3_bucket_models, key="latest.json")
    prefix = str(latest.get("snapshot_prefix", "")).strip()
    if not prefix:
        raise RuntimeError("Latest model metadata is missing snapshot_prefix")
    paths = snapshot_paths_for_prefix(prefix)
    load_manifest(storage, paths.manifest_key)
    if not snapshot_exists(storage, paths):
        raise RuntimeError("Incomplete snapshot")
    return paths
