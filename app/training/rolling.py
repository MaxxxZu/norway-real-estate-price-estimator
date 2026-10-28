from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from app.clients.api_client import ApiClient
from app.config import settings
from app.storage.s3 import S3Storage
from app.training.dataset import NORMALIZATION_VERSION, TrainingIdentity, build_trainable_dataset
from app.training.fetch import (
    FetchConfig,
    build_properties,
    build_rows,
    fetch_estimation_params,
    fetch_turnovers,
    normalize_turnovers,
)
from app.training.snapshots import (
    RawSnapshotPaths,
    SnapshotPaths,
    canonical_json,
    canonical_rows,
    iter_jsonl_rows,
    load_manifest,
    raw_snapshot_exists,
    resolve_raw_snapshot,
    semantic_fingerprint,
    snapshot_exists,
    snapshot_paths_for_prefix,
    upload_raw_snapshot,
    upload_snapshots_with_prefix,
)
from app.training.window import shift_months


@dataclass(frozen=True)
class MonthRange:
    start: date
    end: date


def _first_day_of_month(d: date) -> date:
    return d.replace(day=1)


def _last_day_of_month(d: date) -> date:
    next_month = shift_months(d.replace(day=1), 1)
    return next_month - timedelta(days=1)


def month_ranges(as_of: date, months: int = 12) -> list[MonthRange]:
    if months <= 0:
        raise ValueError("months must be > 0")

    end_month_start = _first_day_of_month(as_of)
    start_month = shift_months(end_month_start, -months)
    ranges: list[MonthRange] = []

    for i in range(months):
        month_start = shift_months(start_month, i)
        month_end = _last_day_of_month(month_start)
        ranges.append(MonthRange(start=month_start, end=month_end))

    return ranges


def ensure_month_snapshot(
    storage: S3Storage,
    api_client: ApiClient,
    start: date,
    end: date,
    cfg: FetchConfig,
    force_fetch: bool = False,
) -> RawSnapshotPaths:
    if not force_fetch:
        paths = resolve_raw_snapshot(storage, start.isoformat(), end.isoformat())
        if paths is not None:
            return paths

    turnovers_raw = fetch_turnovers(api_client, start, end, cfg)
    turnovers = normalize_turnovers(turnovers_raw)
    cadastral_unit_ids = [t["cadastral_unit_ids"][0] for t in turnovers]
    properties = build_properties(api_client, cadastral_unit_ids)
    estimation_params = fetch_estimation_params(api_client, properties)
    rows_raw = build_rows(turnovers, properties, estimation_params)

    manifest = {
        "period": {"start_date": start.isoformat(), "end_date": end.isoformat()},
        "counts": {
            "turnovers_raw": len(turnovers_raw),
            "turnovers_normalized": len(turnovers),
            "cadastral_unit_ids": len(cadastral_unit_ids),
            "properties_matched": len(properties),
            "rows_raw": len(rows_raw),
        },
        "source": "monthly_snapshot",
    }

    return upload_raw_snapshot(
        storage=storage,
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        raw_rows=rows_raw,
        manifest=manifest,
    )


def dedupe_latest_by_property_id(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Input must be normalized. Same-day conflicts use canonical row order."""
    latest = {}
    for row in sorted(rows, key=lambda r: (r["turnover_date"], canonical_json(r))):
        latest[row["id"]] = row
    return canonical_rows(list(latest.values()))


def build_rolling_snapshot(
    storage: S3Storage,
    month_snapshots: list[RawSnapshotPaths],
    as_of: date,
    months: int = 12,
) -> tuple[SnapshotPaths, dict[str, Any]]:
    if months <= 0:
        raise ValueError("months must be > 0")
    window_id = (
        f"{shift_months(as_of.replace(day=1), -months).isoformat()}_"
        f"{(as_of.replace(day=1) - timedelta(days=1)).isoformat()}"
    )
    sources = []
    all_rows = []
    for snap in sorted(month_snapshots, key=lambda item: item.prefix):
        if not raw_snapshot_exists(storage, snap):
            raise RuntimeError("Incomplete or incompatible source snapshot")
        source_manifest = load_manifest(storage, snap.manifest_key)
        rows = list(iter_jsonl_rows(storage, snap.raw_rows_key))
        if semantic_fingerprint(rows) != source_manifest["raw_fingerprint"]:
            raise RuntimeError("Source snapshot fingerprint mismatch")
        sources.append(
            {
                "start_date": snap.start_date,
                "end_date": snap.end_date,
                "raw_rows_key": snap.raw_rows_key,
                "manifest_key": snap.manifest_key,
                "revision": source_manifest["revision"],
                "raw_fingerprint": source_manifest["raw_fingerprint"],
                "normalization_version": source_manifest["normalization_version"],
            }
        )
        all_rows.extend(rows)
    identity = semantic_fingerprint(
        [
            {
                "window": window_id,
                "months": months,
                "normalization_version": NORMALIZATION_VERSION,
                "sources": sources,
            }
        ]
    )
    index_key = f"snapshots/v1/rolling/{window_id}/identities/{identity}.json"
    if storage.exists(settings.s3_bucket_snapshots, index_key):
        ref = storage.get_json(bucket=settings.s3_bucket_snapshots, key=index_key)
        existing_paths = snapshot_paths_for_prefix(ref["prefix"])
        if snapshot_exists(storage, existing_paths):
            existing_manifest = load_manifest(storage, existing_paths.manifest_key)
            if existing_manifest.get("dataset_identity") != identity:
                raise RuntimeError("Snapshot discovery identity mismatch")
            return existing_paths, existing_manifest
    # Each attempt owns its files, including concurrent attempts of the same identity.
    prefix = f"snapshots/v1/rolling/{window_id}/revisions/{uuid4().hex}"
    identity_rows = []
    identity_drops = Counter()
    for row in all_rows:
        try:
            identity_row = TrainingIdentity.model_validate(row).model_dump(mode="json")
        except ValidationError as exc:
            field = ".".join(map(str, exc.errors()[0]["loc"])) or "row"
            identity_drops[f"invalid_identity:{field}"] += 1
            continue
        identity_rows.append({**row, **identity_row})
    deduped_rows = dedupe_latest_by_property_id(identity_rows)
    dataset_result = build_trainable_dataset(deduped_rows)

    window_start = shift_months(as_of.replace(day=1), -months)
    window_end = as_of.replace(day=1) - timedelta(days=1)

    manifest = {
        "period": {"start_date": window_start.isoformat(), "end_date": window_end.isoformat()},
        "window": {"months": months, "as_of": as_of.isoformat()},
        "counts": {
            "rows_raw_total": len(all_rows),
            "rows_raw_deduped": len(deduped_rows),
            "rows_trainable": len(dataset_result.trainable_rows),
        },
        "dropped_reasons": dict(
            sorted((identity_drops + Counter(dataset_result.dropped_reasons)).items())
        ),
        "dataset_identity": identity,
        "source_months": sources,
    }

    paths = upload_snapshots_with_prefix(
        storage=storage,
        prefix=prefix,
        raw_rows=all_rows,
        trainable_rows=dataset_result.trainable_rows,
        manifest=manifest,
    )

    storage.put_json(
        bucket=settings.s3_bucket_snapshots, key=index_key, obj={"prefix": paths.prefix}
    )
    return paths, load_manifest(storage, paths.manifest_key)
