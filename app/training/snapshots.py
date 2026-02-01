import json
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import pandas as pd

from app.config import settings
from app.storage.s3 import S3Storage, S3StorageError


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


def snapshot_paths(start_date: str, end_date: str) -> SnapshotPaths:
    prefix = f"snapshots/{start_date}_{end_date}"
    return SnapshotPaths(
        prefix=prefix,
        raw_rows_key=f"{prefix}/rows_raw.jsonl",
        dataset_key=f"{prefix}/dataset.parquet",
        manifest_key=f"{prefix}/manifest.json",
    )


def snapshot_paths_for_prefix(prefix: str) -> SnapshotPaths:
    return SnapshotPaths(
        prefix=prefix,
        raw_rows_key=f"{prefix}/rows_raw.jsonl",
        dataset_key=f"{prefix}/dataset.parquet",
        manifest_key=f"{prefix}/manifest.json",
    )


def raw_snapshot_paths(start_date: str, end_date: str) -> RawSnapshotPaths:
    prefix = f"snapshots/{start_date}_{end_date}"
    return RawSnapshotPaths(
        prefix=prefix,
        raw_rows_key=f"{prefix}/rows_raw.jsonl",
        manifest_key=f"{prefix}/manifest.json",
        start_date=start_date,
        end_date=end_date,
    )


def upload_jsonl(storage: S3Storage, bucket: str, key: str, rows: list[dict[str, Any]]) -> None:
    lines = [json.dumps(r, ensure_ascii=False) for r in rows]
    data = ("\n".join(lines) + "\n").encode("utf-8")
    storage.put_bytes(bucket=bucket, key=key, data=data, content_type="application/x-ndjson")


def upload_parquet(storage: S3Storage, bucket: str, key: str, rows: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(rows)
    buf = df.to_parquet(index=False)
    if isinstance(buf, bytes):
        data = buf
    else:
        data = bytes(buf)
    storage.put_bytes(bucket=bucket, key=key, data=data, content_type="application/octet-stream")


def upload_manifest(storage: S3Storage, bucket: str, key: str, manifest: dict[str, Any]) -> None:
    storage.put_json(bucket=bucket, key=key, obj=manifest)


def load_jsonl_rows(storage: S3Storage, key: str) -> list[dict[str, Any]]:
    raw = storage.get_bytes(bucket=settings.s3_bucket_snapshots, key=key)
    rows: list[dict[str, Any]] = []
    for line in raw.decode("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def upload_snapshots(
    storage: S3Storage,
    start_date: str,
    end_date: str,
    raw_rows: list[dict[str, Any]],
    trainable_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> SnapshotPaths:
    paths = snapshot_paths(start_date, end_date)
    _upload_snapshot_files(
        storage=storage,
        paths=paths,
        raw_rows=raw_rows,
        trainable_rows=trainable_rows,
        manifest=manifest,
    )
    return paths


def upload_snapshots_with_prefix(
    storage: S3Storage,
    prefix: str,
    raw_rows: list[dict[str, Any]],
    trainable_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> SnapshotPaths:
    paths = snapshot_paths_for_prefix(prefix)
    _upload_snapshot_files(
        storage=storage,
        paths=paths,
        raw_rows=raw_rows,
        trainable_rows=trainable_rows,
        manifest=manifest,
    )
    return paths


def upload_raw_snapshot(
    storage: S3Storage,
    start_date: str,
    end_date: str,
    raw_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> RawSnapshotPaths:
    paths = raw_snapshot_paths(start_date, end_date)
    _upload_raw_snapshot_files(
        storage=storage,
        paths=paths,
        raw_rows=raw_rows,
        manifest=manifest,
    )
    return paths


def snapshot_exists(storage: S3Storage, paths: SnapshotPaths) -> bool:
    return storage.exists(settings.s3_bucket_snapshots, paths.dataset_key) and storage.exists(
        settings.s3_bucket_snapshots, paths.manifest_key
    )


def raw_snapshot_exists(storage: S3Storage, paths: RawSnapshotPaths) -> bool:
    return storage.exists(settings.s3_bucket_snapshots, paths.raw_rows_key) and storage.exists(
        settings.s3_bucket_snapshots, paths.manifest_key
    )


def _upload_snapshot_files(
    storage: S3Storage,
    paths: SnapshotPaths,
    raw_rows: list[dict[str, Any]],
    trainable_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    upload_jsonl(storage, settings.s3_bucket_snapshots, paths.raw_rows_key, raw_rows)
    upload_parquet(storage, settings.s3_bucket_snapshots, paths.dataset_key, trainable_rows)
    upload_manifest(storage, settings.s3_bucket_snapshots, paths.manifest_key, manifest)


def _upload_raw_snapshot_files(
    storage: S3Storage,
    paths: RawSnapshotPaths,
    raw_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    upload_jsonl(storage, settings.s3_bucket_snapshots, paths.raw_rows_key, raw_rows)
    upload_manifest(storage, settings.s3_bucket_snapshots, paths.manifest_key, manifest)


def load_manifest(storage: S3Storage, manifest_key: str) -> dict[str, Any]:
    try:
        return storage.get_json(bucket=settings.s3_bucket_snapshots, key=manifest_key)
    except S3StorageError as e:
        raise RuntimeError(
            f"Latest snapshot is not initialized (missing {manifest_key}). "
            "Run training with --force-fetch to create it."
        ) from e


def load_trainable_rows_from_parquet(storage: S3Storage, dataset_key: str) -> list[dict[str, Any]]:
    raw = storage.get_bytes(bucket=settings.s3_bucket_snapshots, key=dataset_key)
    df = pd.read_parquet(BytesIO(raw))
    return df.to_dict(orient="records")


def fetch_latest_snapshot_ref(storage: S3Storage) -> SnapshotPaths:
    latest = storage.get_json(bucket=settings.s3_bucket_models, key="latest.json")
    snapshot_prefix = str(latest.get("snapshot_prefix", "")).strip()
    if not snapshot_prefix:
        raise RuntimeError("Latest model metadata is missing snapshot_prefix")

    return SnapshotPaths(
        prefix=snapshot_prefix,
        raw_rows_key=f"{snapshot_prefix}/rows_raw.jsonl",
        dataset_key=f"{snapshot_prefix}/dataset.parquet",
        manifest_key=f"{snapshot_prefix}/manifest.json",
    )
