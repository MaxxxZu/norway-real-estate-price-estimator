import json
from datetime import date

from app.training.rolling import (
    MonthRange,
    build_rolling_snapshot,
    dedupe_latest_by_property_id,
    month_ranges,
)
from app.training.snapshots import RawSnapshotPaths, SnapshotPaths


class FakeStorage:
    def __init__(self, data: dict[str, bytes]):
        self._data = data

    def get_bytes(self, bucket: str, key: str) -> bytes:
        return self._data[key]


def _jsonl(rows: list[dict]) -> bytes:
    return ("\n".join(json.dumps(r) for r in rows) + "\n").encode("utf-8")


def test_month_ranges_12_months():
    as_of = date(2026, 2, 15)
    ranges = month_ranges(as_of, months=12)
    assert len(ranges) == 12
    assert ranges[0] == MonthRange(start=date(2025, 2, 1), end=date(2025, 2, 28))
    assert ranges[-1] == MonthRange(start=date(2026, 1, 1), end=date(2026, 1, 31))


def test_dedupe_latest_by_property_id_keeps_newest():
    rows = [
        {"id": 1, "turnover_date": "2026-01-10", "price": 1},
        {"id": 1, "turnover_date": "2026-01-20", "price": 2},
        {"id": 2, "turnover_date": "2026-01-05", "price": 3},
    ]
    deduped = dedupe_latest_by_property_id(rows)
    by_id = {r["id"]: r for r in deduped}
    assert by_id[1]["price"] == 2
    assert by_id[2]["price"] == 3


def test_build_rolling_snapshot_manifest(monkeypatch):
    month1 = RawSnapshotPaths(
        prefix="snapshots/2025-01-01_2025-01-31",
        raw_rows_key="snapshots/2025-01-01_2025-01-31/rows_raw.jsonl",
        manifest_key="snapshots/2025-01-01_2025-01-31/manifest.json",
        start_date="2025-01-01",
        end_date="2025-01-31",
    )
    month2 = RawSnapshotPaths(
        prefix="snapshots/2025-02-01_2025-02-28",
        raw_rows_key="snapshots/2025-02-01_2025-02-28/rows_raw.jsonl",
        manifest_key="snapshots/2025-02-01_2025-02-28/manifest.json",
        start_date="2025-02-01",
        end_date="2025-02-28",
    )
    storage = FakeStorage(
        {
            month1.raw_rows_key: _jsonl(
                [
                    {
                        "id": 1,
                        "turnover_date": "2025-01-10",
                        "price": 100,
                        "bra": 10,
                        "total_area": 12,
                    },
                    {
                        "id": 2,
                        "turnover_date": "2025-01-15",
                        "price": 200,
                        "bra": 20,
                        "total_area": 25,
                    },
                ]
            ),
            month2.raw_rows_key: _jsonl(
                [
                    {
                        "id": 1,
                        "turnover_date": "2025-02-05",
                        "price": 150,
                        "bra": 10,
                        "total_area": 12,
                    },
                    {
                        "id": 3,
                        "turnover_date": "2025-02-10",
                        "price": 300,
                        "bra": 30,
                        "total_area": 35,
                    },
                ]
            ),
        }
    )

    captured: dict[str, object] = {}

    def fake_upload_snapshots_with_prefix(
        storage, prefix, raw_rows, trainable_rows, manifest
    ) -> SnapshotPaths:
        captured["manifest"] = manifest
        return SnapshotPaths(
            prefix=prefix,
            raw_rows_key=f"{prefix}/rows_raw.jsonl",
            dataset_key=f"{prefix}/dataset.parquet",
            manifest_key=f"{prefix}/manifest.json",
        )

    monkeypatch.setattr(
        "app.training.rolling.upload_snapshots_with_prefix", fake_upload_snapshots_with_prefix
    )

    _, manifest = build_rolling_snapshot(
        storage=storage,
        month_snapshots=[month1, month2],
        as_of=date(2025, 3, 1),
        months=2,
    )

    assert manifest["counts"]["rows_raw_total"] == 4
    assert manifest["counts"]["rows_raw_deduped"] == 3
    assert len(manifest["source_months"]) == 2
    assert captured["manifest"] == manifest
