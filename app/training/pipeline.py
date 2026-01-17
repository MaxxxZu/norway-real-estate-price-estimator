from datetime import date
from typing import Any

from app.clients.api_client import ApiClient
from app.storage.s3 import S3Storage
from app.training.dataset import build_trainable_dataset, DatasetBuildResult
from app.training.fetch import (
    FetchConfig,
    build_properties,
    build_rows,
    fetch_estimation_params,
    fetch_turnovers,
    normalize_turnovers,
)
from app.training.modeling import train_and_evaluate
from app.training.publish import update_latest_json, upload_model_artifacts
from app.training.snapshots import snapshot_paths, upload_snapshots
from app.training.versioning import make_model_version


def _add_snapshot(
    start_date: date,
    end_date: date,
    turnovers_raw: list[dict],
    turnovers: list[dict],
    cadastral_unit_ids: list[str],
    properties: list[dict],
    rows_raw: list[dict],
    dataset_result: DatasetBuildResult,
    dry_run: bool,
    train: bool,
    publish: bool,
) -> list[dict, Any]:
    storage = S3Storage()
    manifest = {
        "period": {"start_date": start_date.isoformat(), "end_date": end_date.isoformat()},
        "counts": {
            "turnovers_raw": len(turnovers_raw),
            "turnovers_normalized": len(turnovers),
            "cadastral_unit_ids": len(cadastral_unit_ids),
            "properties_matched": len(properties),
            "rows_raw": len(rows_raw),
            "rows_trainable": len(dataset_result.trainable_rows),
        },
        "dropped_reasons": dataset_result.dropped_reasons,
        "dry_run": bool(dry_run),
        "train": bool(train),
        "publish": bool(publish),
    }

    paths = upload_snapshots(
        storage=storage,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        raw_rows=rows_raw,
        trainable_rows=dataset_result.trainable_rows,
        manifest=manifest,
    )

    return manifest, paths


def _publish_model(
    start_date: date,
    end_date: date,
    manifest: dict[str, Any],
    train_res: Any
) -> dict:
    storage = S3Storage()

    model_version = make_model_version()
    snapshot_prefix = snapshot_paths(start_date.isoformat(), end_date.isoformat()).prefix

    training_manifest = {
        "model_version": model_version,
        "snapshot_prefix": snapshot_prefix,
        "period": manifest["period"],
        "counts": manifest["counts"],
        "dropped_reasons": manifest["dropped_reasons"],
        "metrics": train_res.metrics,
    }

    keys = upload_model_artifacts(
        storage=storage,
        model_version=model_version,
        pipeline=train_res.pipeline,
        metrics=train_res.metrics,
        feature_schema=train_res.feature_schema,
        training_manifest=training_manifest,
    )

    update_latest_json(
        storage=storage,
        model_version=model_version,
        artifact_key=keys["model_key"],
        snapshot_prefix=snapshot_prefix,
    )

    return {
        "model_version": model_version,
        "model_key": keys["model_key"],
        "latest_key": "latest.json",
    }


def run_training_pipeline(
    *,
    start_date: date,
    end_date: date,
    dry_run: bool,
    train: bool,
    publish: bool,
) -> dict:
    if publish and not train:
        raise ValueError("publish=True requires train=True")
    if dry_run and train:
        raise ValueError("dry_run and train are mutually exclusive")

    if end_date < start_date:
        raise ValueError("end_date must be >= start_date")

    api = ApiClient()
    cfg = FetchConfig()

    turnovers_raw = fetch_turnovers(api, start_date, end_date, cfg)
    turnovers = normalize_turnovers(turnovers_raw)

    cadastral_unit_ids = [t["cadastral_unit_ids"][0] for t in turnovers]
    properties = build_properties(api, cadastral_unit_ids)

    estimation_params = fetch_estimation_params(api, properties)
    rows_raw = build_rows(turnovers, properties, estimation_params)

    dataset_result = build_trainable_dataset(rows_raw)
    manifest, paths = _add_snapshot(
        start_date,
        end_date,
        turnovers_raw,
        turnovers,
        cadastral_unit_ids,
        properties,
        rows_raw,
        dataset_result,
        dry_run,
        train,
        publish,
    )

    result: dict = {
        "snapshots": {
            "raw_rows_key": paths.raw_rows_key,
            "dataset_key": paths.dataset_key,
            "manifest_key": paths.manifest_key,
        },
        "manifest": manifest,
    }
    if dry_run:
        return result

    train_res = train_and_evaluate(dataset_result.trainable_rows)
    result["metrics"] = train_res.metrics
    if not publish:
        return result

    result["published"] = _publish_model(start_date, end_date, manifest, train_res)

    return result
