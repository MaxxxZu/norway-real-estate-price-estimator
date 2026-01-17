from typing import Any
import argparse
from datetime import date

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


DATE_FORMAT = "%Y-%m-%d"


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Training pipeline: fetch -> dataset -> train -> publish"
    )
    parser.add_argument(
        "--start-date",
        required=True,
        type=_parse_date,
        help=(f"Start date in format {DATE_FORMAT} (example: 2022-04-01). ")
    )
    parser.add_argument(
        "--end-date",
        required=True,
        type=_parse_date,
        help=(f"End date in format {DATE_FORMAT} (example: 2022-04-01). ")
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only fetch+snapshot (no training/publish model)"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train a sklearn model on the dataset"
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Publish model artifacts and update latest.json (requires --train)"
    )

    return parser


def _add_snapshot(
    start_date: date,
    end_date: date,
    turnovers_raw: list[dict],
    turnovers: list[dict],
    cadastral_unit_ids: list[str],
    properties: list[dict],
    rows_raw: list[dict],
    dataset_result: DatasetBuildResult,
    args
) -> dict[str, Any]:
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
        "dry_run": bool(args.dry_run),
        "train": bool(args.train),
        "publish": bool(args.publish),
    }

    paths = upload_snapshots(
        storage=storage,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        raw_rows=rows_raw,
        trainable_rows=dataset_result.trainable_rows,
        manifest=manifest,
    )

    print("OK: snapshots uploaded")
    print(f"- {paths.raw_rows_key}")
    print(f"- {paths.dataset_key}")
    print(f"- {paths.manifest_key}")

    return manifest


def _publish_model(
    start_date: date,
    end_date: date,
    manifest: dict[str, Any],
    train_res: Any
) -> None:
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

    print("OK: model artifacts published")
    print(f"- {keys['model_key']}")
    print("OK: latest.json updated")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.dry_run and not args.train:
        args.dry_run = True

    if args.dry_run and args.train:
        raise SystemExit("Choose either --dry-run or --train (not both).")

    if args.publish and not args.train:
        raise SystemExit("--publish requires --train.")

    start_date = args.start_date
    end_date = args.end_date
    if end_date < start_date:
        raise SystemExit("end-date must be >= start-date")

    api = ApiClient()
    cfg = FetchConfig()

    turnovers_raw = fetch_turnovers(api, start_date, end_date, cfg)
    turnovers = normalize_turnovers(turnovers_raw)

    cadastral_unit_ids = [t["cadastral_unit_ids"][0] for t in turnovers]
    properties = build_properties(api, cadastral_unit_ids)

    estimation_params = fetch_estimation_params(api, properties)
    rows_raw = build_rows(turnovers, properties, estimation_params)

    dataset_result = build_trainable_dataset(rows_raw)
    manifest = _add_snapshot(
        start_date,
        end_date,
        turnovers_raw,
        turnovers,
        cadastral_unit_ids,
        properties,
        rows_raw,
        dataset_result,
        args,
    )

    if args.dry_run:
        return

    # Train
    train_res = train_and_evaluate(dataset_result.trainable_rows)
    print("OK: training finished")
    print("Metrics:", train_res.metrics)

    if not args.publish:
        print(
            "NOTE: --publish not specified, model artifacts were not uploaded "
            "and latest.json not updated."
        )
        return

    _publish_model(start_date, end_date, manifest, train_res)


if __name__ == "__main__":
    main()
