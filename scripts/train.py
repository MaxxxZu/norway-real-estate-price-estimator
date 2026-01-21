import argparse
from datetime import date

from app.training.pipeline import run_training_pipeline
from app.training.window import window_start


DATE_FORMAT = "%Y-%m-%d"


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Training pipeline: fetch -> dataset -> train -> publish"
    )
    parser.add_argument(
        "--end-date",
        required=True,
        type=_parse_date,
        help=(f"End date in format {DATE_FORMAT} (example: 2022-04-01). ")
    )
    parser.add_argument(
        "--start-date",
        required=False,
        type=_parse_date,
        help=(f"Optional start date in format {DATE_FORMAT} (example: 2022-04-01). ")
    )
    parser.add_argument(
        "--window-months",
        type=int,
        default=12,
        help="Trailing window size in months (used when --start-date is omitted). Default: 12.",
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


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.dry_run and not args.train:
        args.dry_run = True

    if args.dry_run and args.train:
        raise SystemExit("Choose either --dry-run or --train (not both).")

    if args.publish and not args.train:
        raise SystemExit("--publish requires --train.")

    end_date = args.end_date
    start_date: date = args.start_date or window_start(end_date, args.window_months)
    if end_date < start_date:
        raise SystemExit("end-date must be >= start-date")

    res = run_training_pipeline(
        start_date=start_date,
        end_date=end_date,
        dry_run=bool(args.dry_run),
        train=bool(args.train),
        publish=bool(args.publish),
    )

    print("OK: snapshots uploaded")
    print(f"- {res['snapshots']['raw_rows_key']}")
    print(f"- {res['snapshots']['dataset_key']}")
    print(f"- {res['snapshots']['manifest_key']}")

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
