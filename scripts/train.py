import argparse
from datetime import date

from app.clients.api_client import ApiClient
from app.storage.s3 import S3Storage
from app.training.dataset import build_trainable_dataset
from app.training.fetch import (
    FetchConfig,
    build_properties,
    build_rows,
    fetch_estimation_params,
    fetch_turnovers,
    normalize_turnovers,
)
from app.training.snapshots import upload_snapshots


DATE_FORMAT = "%Y-%m-%d"


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Training pipeline (dry-run): fetch -> dataset -> snapshots"
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

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    start_date = args.start_date
    end_date = args.end_date
    if end_date < start_date:
        raise SystemExit("end-date must be >= start-date")

    api = ApiClient()
    storage = S3Storage()
    cfg = FetchConfig()

    turnovers_raw = fetch_turnovers(api, start_date, end_date, cfg)
    turnovers = normalize_turnovers(turnovers_raw)

    cadastral_unit_ids = [t["cadastral_unit_ids"][0] for t in turnovers]
    properties = build_properties(api, cadastral_unit_ids)

    estimation_params = fetch_estimation_params(api, properties)
    rows_raw = build_rows(turnovers, properties, estimation_params)

    dataset_result = build_trainable_dataset(rows_raw)

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

    if not args.dry_run:
        print("NOTE: training/publish is not implemented yet (this PR is dry-run only).")


if __name__ == "__main__":
    main()
