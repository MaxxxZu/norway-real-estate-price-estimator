from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from app.clients.api_client import ApiClient
from app.storage.s3 import S3Storage
from app.training.dataset import DatasetBuildResult, build_trainable_dataset
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
    load_jsonl_rows,
    raw_snapshot_exists,
    raw_snapshot_paths,
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
    paths = raw_snapshot_paths(start.isoformat(), end.isoformat())
    if not force_fetch and raw_snapshot_exists(storage, paths):
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


def _parse_turnover_date(value: Any) -> datetime:
    s = str(value)
    if "T" in s:
        # 2026-01-17T07:12:09.978Z
        try:
            return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%fZ")
        except ValueError:
            # fallback: strip Z or millis
            return datetime.fromisoformat(s.replace("Z", ""))
    return datetime.fromisoformat(s)


def dedupe_latest_by_property_id(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[int, dict[str, Any]] = {}
    latest_dt: dict[int, date] = {}

    for row in rows:
        prop_id = int(row.get("id") or row.get("property_id") or row.get("remote_id"))
        row_dt = _parse_turnover_date(row.get("turnover_date"))
        if row_dt is None:
            continue

        prev_dt = latest_dt.get(prop_id)
        if prev_dt is None or row_dt >= prev_dt:
            latest[prop_id] = row
            latest_dt[prop_id] = row_dt

    return list(latest.values())


def build_rolling_snapshot(
    storage: S3Storage,
    month_snapshots: list[RawSnapshotPaths],
    as_of: date,
    months: int = 12,
) -> tuple[SnapshotPaths, dict[str, Any]]:
    window_id = (
        f"{shift_months(as_of.replace(day=1), -months).isoformat()}_"
        f"{(as_of.replace(day=1) - timedelta(days=1)).isoformat()}"
    )
    prefix = f"snapshots/rolling_12m/{window_id}"

    all_rows: list[dict[str, Any]] = []
    for snap in month_snapshots:
        all_rows.extend(load_jsonl_rows(storage, snap.raw_rows_key))

    deduped_rows = dedupe_latest_by_property_id(all_rows)
    dataset_result: DatasetBuildResult = build_trainable_dataset(deduped_rows)

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
        "dropped_reasons": dataset_result.dropped_reasons,
        "source_months": [
            {
                "start_date": snap.start_date,
                "end_date": snap.end_date,
                "raw_rows_key": snap.raw_rows_key,
                "manifest_key": snap.manifest_key,
            }
            for snap in month_snapshots
        ],
    }

    paths = upload_snapshots_with_prefix(
        storage=storage,
        prefix=prefix,
        raw_rows=deduped_rows,
        trainable_rows=dataset_result.trainable_rows,
        manifest=manifest,
    )

    return paths, manifest
