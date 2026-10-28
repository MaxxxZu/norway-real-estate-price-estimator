import copy
import json
from datetime import date

import pytest

from app.storage.s3 import S3StorageError
from app.training.dataset import build_trainable_dataset
from app.training.rolling import (
    MonthRange,
    build_rolling_snapshot,
    dedupe_latest_by_property_id,
    ensure_month_snapshot,
    month_ranges,
)
from app.training.snapshots import (
    load_manifest,
    load_trainable_rows_from_parquet,
    raw_latest_key,
    raw_snapshot_exists,
    resolve_raw_snapshot,
    semantic_fingerprint,
    snapshot_exists,
    snapshot_paths_for_prefix,
    upload_raw_snapshot,
    upload_snapshots_with_prefix,
)


class FakeStorage:
    def __init__(self):
        self.data = {}
        self.writes = []
        self.fail_suffix = None

    def get_bytes(self, bucket, key):
        if key not in self.data:
            raise S3StorageError("Synthetic missing object")
        return self.data[key]

    def get_json(self, bucket, key):
        return json.loads(self.get_bytes(bucket, key))

    def put_bytes(self, bucket, key, data, content_type=None):
        if self.fail_suffix and key.endswith(self.fail_suffix):
            raise S3StorageError("Synthetic interrupted write")
        self.writes.append(key)
        self.data[key] = data

    def put_json(self, bucket, key, obj):
        self.put_bytes(bucket, key, json.dumps(obj).encode())

    def iter_lines(self, bucket, key):
        yield from self.get_bytes(bucket, key).decode().splitlines()

    def exists(self, bucket, key):
        return key in self.data


def row(**updates):
    return {
        "id": 1,
        "turnover_date": "2025-01-10",
        "price": 1000000,
        "realestate_type": "enebolig",
        "municipality_number": "0301",
        "lat": 59.9,
        "lon": 10.7,
        "built_year": 2000,
        "bra": 50,
        "total_area": 55,
        **updates,
    }


def source(storage, rows=None):
    return upload_raw_snapshot(storage, "2025-01-01", "2025-01-31", rows or [row()], {})


def merge(storage, sources):
    return build_rolling_snapshot(storage, sources, date(2025, 2, 1), months=1)


def test_month_ranges_12_months():
    ranges = month_ranges(date(2026, 2, 15), months=12)
    assert len(ranges) == 12
    assert ranges[0] == MonthRange(date(2025, 2, 1), date(2025, 2, 28))
    assert ranges[-1] == MonthRange(date(2026, 1, 1), date(2026, 1, 31))


def test_dedupe_latest_by_property_id_keeps_newest_and_resolves_ties():
    rows = [
        row(),
        row(turnover_date="2025-01-20", price=2),
        row(turnover_date="2025-01-20", price=3),
        row(id=2),
    ]
    expected = dedupe_latest_by_property_id(rows)
    assert expected == dedupe_latest_by_property_id(list(reversed(rows)))
    assert {r["id"]: r for r in expected}[1]["turnover_date"] == "2025-01-20"


def test_semantic_fingerprint_order_and_types():
    assert semantic_fingerprint([{"a": 1, "b": 2.0}, {"a": 3}]) == semantic_fingerprint(
        [{"a": 3.0}, {"b": 2, "a": 1.0}]
    )
    assert semantic_fingerprint([row()]) != semantic_fingerprint([row(), row()])
    assert semantic_fingerprint([row(price=float("nan"))]) == semantic_fingerprint(
        [row(price=float("nan"))]
    )


def test_build_rolling_snapshot_manifest_and_idempotence():
    storage = FakeStorage()
    snap = source(storage, [row(), row(id=2), row(turnover_date="2025-01-20")])
    paths, manifest = merge(storage, [snap])
    assert manifest["counts"] == {"rows_raw_total": 3, "rows_raw_deduped": 2, "rows_trainable": 2}
    assert manifest["source_months"][0]["revision"] == snap.prefix.rsplit("/", 1)[-1]
    assert manifest["complete"] is True
    assert storage.writes[-2] == paths.manifest_key
    assert "/identities/" in storage.writes[-1]
    previous_writes = list(storage.writes)
    assert merge(storage, [snap]) == (paths, manifest)
    assert storage.writes == previous_writes
    loaded = load_trainable_rows_from_parquet(storage, paths.dataset_key)
    assert len(loaded) == 2
    assert loaded[0]["municipality_number"] == 301
    assert loaded[0]["floor"] is None
    assert semantic_fingerprint(loaded) == manifest["fingerprint"]


def test_raw_completion_failure_and_retry():
    storage = FakeStorage()
    storage.fail_suffix = "manifest.json"
    with pytest.raises(S3StorageError):
        source(storage)
    assert resolve_raw_snapshot(storage, "2025-01-01", "2025-01-31") is None
    assert all(not key.endswith("latest_raw.json") for key in storage.data)
    previous = copy.deepcopy(storage.data)
    storage.fail_suffix = None
    snap = source(storage)
    assert raw_snapshot_exists(storage, snap)
    assert all(storage.data[key] == value for key, value in previous.items())
    assert storage.writes[-2:] == [snap.manifest_key, raw_latest_key("2025-01-01", "2025-01-31")]


@pytest.mark.parametrize("suffix", ["dataset.parquet", "manifest.json"])
def test_incomplete_dataset_not_used_and_retry(suffix):
    storage = FakeStorage()
    snap = source(storage)
    storage.fail_suffix = suffix
    with pytest.raises(S3StorageError):
        merge(storage, [snap])
    prefixes = {key.rsplit("/", 1)[0] for key in storage.data if "/rolling/" in key}
    assert prefixes
    for prefix in prefixes:
        assert not snapshot_exists(storage, snapshot_paths_for_prefix(prefix))
    storage.fail_suffix = None
    paths, _ = merge(storage, [snap])
    assert snapshot_exists(storage, paths)


def test_force_fetch_invalidates_merge_preserves_lineage(monkeypatch):
    storage = FakeStorage()
    old_source = source(storage)
    old_paths, old_manifest = merge(storage, [old_source])
    old_data = copy.deepcopy(storage.data)
    calls = []
    monkeypatch.setattr("app.training.rolling.fetch_turnovers", lambda *a: calls.append(1) or [])
    monkeypatch.setattr("app.training.rolling.build_properties", lambda *a: {})
    monkeypatch.setattr("app.training.rolling.fetch_estimation_params", lambda *a: {})
    monkeypatch.setattr("app.training.rolling.build_rows", lambda *a: [row(price=2000000)])
    from app.training.fetch import FetchConfig

    kwargs = {
        "storage": storage,
        "api_client": object(),
        "start": date(2025, 1, 1),
        "end": date(2025, 1, 31),
        "cfg": FetchConfig(),
    }
    assert ensure_month_snapshot(**kwargs) == old_source
    assert calls == []
    new_source = ensure_month_snapshot(**kwargs, force_fetch=True)
    new_paths, new_manifest = merge(storage, [new_source])
    assert new_paths.prefix != old_paths.prefix
    assert new_manifest["fingerprint"] != old_manifest["fingerprint"]
    assert calls == [1]
    assert merge(storage, [old_source]) == (old_paths, old_manifest)
    # Only the discovery pointer may change; every historical object is unchanged.
    pointer = raw_latest_key("2025-01-01", "2025-01-31")
    assert all(storage.data[key] == value for key, value in old_data.items() if key != pointer)


def test_identical_refetch_still_creates_new_lineage():
    storage = FakeStorage()
    first, second = source(storage), source(storage)
    paths1, manifest1 = merge(storage, [first])
    paths2, manifest2 = merge(storage, [second])
    assert paths1 != paths2
    assert manifest1["fingerprint"] == manifest2["fingerprint"]


def test_source_order_does_not_change_identity():
    storage = FakeStorage()
    first, second = source(storage), source(storage, [row(id=2)])
    assert merge(storage, [first, second]) == merge(storage, [second, first])


def test_immutable_completed_snapshot_conflict():
    storage = FakeStorage()
    paths, _ = merge(storage, [source(storage)])
    previous = copy.deepcopy(storage.data)
    with pytest.raises(RuntimeError, match="conflict"):
        upload_snapshots_with_prefix(
            storage,
            paths.prefix,
            [row(price=2)],
            build_trainable_dataset([row(price=2)]).trainable_rows,
            {},
        )
    assert storage.data == previous


def test_corrupt_source_and_missing_completed_data_rejected():
    storage = FakeStorage()
    snap = source(storage)
    storage.data[snap.raw_rows_key] = b"{}\n"
    with pytest.raises(RuntimeError, match="fingerprint"):
        merge(storage, [snap])
    del storage.data[snap.manifest_key]
    with pytest.raises(RuntimeError, match="Incomplete"):
        merge(storage, [snap])


def test_legacy_snapshot_rejected_without_modification():
    storage = FakeStorage()
    storage.put_json("", "legacy/manifest.json", {"counts": {"rows_trainable": 1}})
    before = copy.deepcopy(storage.data)
    with pytest.raises(RuntimeError, match="Legacy"):
        load_manifest(storage, "legacy/manifest.json")
    assert not snapshot_exists(storage, snapshot_paths_for_prefix("legacy"))
    assert storage.data == before


def test_invalid_latest_transaction_does_not_fall_back_to_older_features():
    storage = FakeStorage()
    snap = source(
        storage, [row(), row(turnover_date="2025-01-20", bra="invalid"), row(id="invalid")]
    )
    paths, manifest = merge(storage, [snap])
    assert manifest["counts"]["rows_trainable"] == 0
    assert manifest["counts"]["rows_raw_deduped"] == 1
    assert manifest["dropped_reasons"] == {"invalid:bra": 1, "invalid_identity:id": 1}
    assert load_trainable_rows_from_parquet(storage, paths.dataset_key) == []


def test_parquet_corruption_detected():
    storage = FakeStorage()
    paths, _ = merge(storage, [source(storage)])
    from app.training.snapshots import upload_parquet

    upload_parquet(
        storage, "", paths.dataset_key, build_trainable_dataset([row(price=2)]).trainable_rows
    )
    with pytest.raises(RuntimeError, match="fingerprint"):
        load_trainable_rows_from_parquet(storage, paths.dataset_key)


def test_missing_completed_dataset_is_rebuilt_without_reusing_partial_revision():
    storage = FakeStorage()
    snap = source(storage)
    old_paths, old_manifest = merge(storage, [snap])
    del storage.data[old_paths.dataset_key]
    paths, manifest = merge(storage, [snap])
    assert paths != old_paths
    assert manifest["dataset_identity"] == old_manifest["dataset_identity"]
    assert old_paths.dataset_key not in storage.data


def test_normalization_version_changes_merge_identity(monkeypatch):
    storage = FakeStorage()
    snap = source(storage)
    paths1, manifest1 = merge(storage, [snap])
    # Change only the merge algorithm version, leaving source format readable.
    monkeypatch.setattr("app.training.rolling.NORMALIZATION_VERSION", 2)
    paths2, manifest2 = merge(storage, [snap])
    assert paths1 != paths2
    assert manifest1["dataset_identity"] != manifest2["dataset_identity"]


def test_failed_raw_refresh_preserves_old_discovery_pointer():
    storage = FakeStorage()
    previous = source(storage)
    storage.fail_suffix = "manifest.json"
    with pytest.raises(S3StorageError):
        source(storage, [row(price=2000000)])
    assert resolve_raw_snapshot(storage, "2025-01-01", "2025-01-31") == previous


def test_discovery_failure_leaves_completed_revision_intact():
    storage = FakeStorage()
    snap = source(storage)
    # Fail after all immutable files have been committed.
    original = storage.put_json

    def fail_index(bucket, key, obj):
        if "/identities/" in key:
            raise S3StorageError("Synthetic discovery failure")
        original(bucket, key, obj)

    storage.put_json = fail_index
    with pytest.raises(S3StorageError):
        merge(storage, [snap])
    completed_keys = [
        key for key in storage.data if "/rolling/" in key and key.endswith("manifest.json")
    ]
    assert len(completed_keys) == 1
    before = copy.deepcopy(storage.data)
    storage.put_json = original
    paths, _ = merge(storage, [snap])
    assert paths.manifest_key != completed_keys[0]
    assert all(storage.data[key] == val for key, val in before.items())


def test_nonfinite_fingerprint_tags_cannot_collide_with_source_objects():
    assert semantic_fingerprint([{"x": float("nan")}]) != semantic_fingerprint(
        [{"x": {"__nonfinite_float__": "nan"}}]
    )


def test_synchronous_revisions_and_repeated_explicit_upload():
    from app.training.snapshots import upload_snapshots

    storage = FakeStorage()
    raw = [row()]
    normalized = build_trainable_dataset(raw).trainable_rows
    manifest = {"period": {"start_date": "2025-01-01", "end_date": "2025-01-31"}}
    first = upload_snapshots(storage, "2025-01-01", "2025-01-31", raw, normalized, manifest)
    before = copy.deepcopy(storage.data)
    second = upload_snapshots(storage, "2025-01-01", "2025-01-31", raw, normalized, manifest)
    assert first != second
    assert all(storage.data[key] == val for key, val in before.items())
    writes = list(storage.writes)
    repeated = upload_snapshots_with_prefix(storage, first.prefix, raw, normalized, manifest)
    assert repeated == first
    assert storage.writes == writes
