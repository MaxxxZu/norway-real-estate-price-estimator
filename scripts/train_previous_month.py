import argparse
from datetime import date, timedelta

from app.training.pipeline import run_training_pipeline


def previous_month_range(today: date) -> tuple[date, date]:
    first_of_this_month = today.replace(day=1)
    end_prev_month = first_of_this_month - timedelta(days=1)
    start_prev_month = end_prev_month.replace(day=1)
    return start_prev_month, end_prev_month


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train & publish for the previous calendar month (cron-friendly)."
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only fetch+snapshot (no training/publish model).",
    )
    p.add_argument(
        "--no-publish",
        action="store_true",
        help="Train but do not publish (do not update latest.json).",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    start_date, end_date = previous_month_range(date.today())
    train = not args.dry_run
    publish = train and (not args.no_publish)

    res = run_training_pipeline(
        start_date=start_date,
        end_date=end_date,
        dry_run=bool(args.dry_run),
        train=train,
        publish=publish,
    )

    print("OK: snapshots uploaded")
    print(f"- {res['snapshots']['raw_rows_key']}")
    print(f"- {res['snapshots']['dataset_key']}")
    print(f"- {res['snapshots']['manifest_key']}")
    print(f"Period: {start_date.isoformat()} -> {end_date.isoformat()}")

    if args.dry_run:
        return

    print("OK: training finished")
    print("Metrics:", res.get("metrics"))

    if publish and "published" in res:
        print("OK: model artifacts published")
        print(f"- {res['published']['model_key']}")
        print("OK: latest.json updated")
    elif train and not publish:
        print("NOTE: publish disabled; latest.json not updated.")


if __name__ == "__main__":
    main()
