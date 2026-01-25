import argparse
from datetime import date

from app.training.pipeline import run_training_pipeline

DATE_FORMAT = "%Y-%m-%d"


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Training pipeline: fetch -> dataset -> train -> publish"
                    "(default uses latest snapshot)"
    )
    parser.add_argument(
        "--start-date",
        type=_parse_date,
        default=None,
        help=f"Start date in format {DATE_FORMAT} (example: 2025-01-01). "
             "Required only with --force-fetch.",
    )
    parser.add_argument(
        "--end-date",
        type=_parse_date,
        default=None,
        help=f"End date in format {DATE_FORMAT} (example: 2025-12-31). "
             "Required only with --force-fetch.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only snapshot (no training/publish). If no --force-fetch, will "
             "only validate that latest snapshot exists.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train a sklearn model on the dataset",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Publish model artifacts and update latest.json (requires --train)",
    )

    parser.add_argument(
        "--force-fetch",
        action="store_true",
        help="Fetch from external API for the given dates, build snapshot "
             "and update snapshots/latest.json.",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.dry_run and not args.train:
        args.dry_run = True

    if args.dry_run and args.train:
        raise SystemExit("Choose either --dry-run or --train (not both).")

    if args.publish and not args.train:
        raise SystemExit("--publish requires --train.")

    if args.force_fetch:
        if args.start_date is None or args.end_date is None:
            raise SystemExit("--force-fetch requires --start-date and --end-date.")
        if args.end_date < args.start_date:
            raise SystemExit("end-date must be >= start-date")

    res = run_training_pipeline(
        start_date=args.start_date,
        end_date=args.end_date,
        dry_run=bool(args.dry_run),
        train=bool(args.train),
        publish=bool(args.publish),
        force_fetch=bool(args.force_fetch),
    )

    print("OK: snapshots")
    print(f"- {res['snapshots']['raw_rows_key']}")
    print(f"- {res['snapshots']['dataset_key']}")
    print(f"- {res['snapshots']['manifest_key']}")
    if "snapshot_latest_key" in res["snapshots"]:
        print(f"- {res['snapshots']['snapshot_latest_key']}")

    if args.dry_run:
        return

    print("OK: training finished")
    print("Metrics:", res.get("metrics"))

    if args.publish and "published" in res:
        print("OK: model artifacts published")
        print(f"- {res['published']['model_key']}")
        print("OK: latest.json updated")
    elif args.train and not args.publish:
        print("NOTE: --publish not specified, latest.json not updated.")


if __name__ == "__main__":
    main()
