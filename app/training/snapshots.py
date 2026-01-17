import json
from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.config import settings
from app.storage.s3 import S3Storage


@dataclass(frozen=True)
class SnapshotPaths:
    prefix: str
    raw_rows_key: str
    dataset_key: str
    manifest_key: str


def snapshot_paths(start_date: str, end_date: str) -> SnapshotPaths:
    prefix = f"snapshots/{start_date}_{end_date}"
    return SnapshotPaths(
        prefix=prefix,
        raw_rows_key=f"{prefix}/rows_raw.jsonl",
        dataset_key=f"{prefix}/dataset.parquet",
        manifest_key=f"{prefix}/manifest.json",
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


def upload_snapshots(
    storage: S3Storage,
    start_date: str,
    end_date: str,
    raw_rows: list[dict[str, Any]],
    trainable_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> SnapshotPaths:
    paths = snapshot_paths(start_date, end_date)

    upload_jsonl(storage, settings.s3_bucket_snapshots, paths.raw_rows_key, raw_rows)
    upload_parquet(storage, settings.s3_bucket_snapshots, paths.dataset_key, trainable_rows)
    upload_manifest(storage, settings.s3_bucket_snapshots, paths.manifest_key, manifest)

    return paths
