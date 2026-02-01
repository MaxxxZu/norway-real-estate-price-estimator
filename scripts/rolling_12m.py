import argparse
from datetime import date

from app.tasks.rolling import trigger_rolling_12m

DATE_FORMAT = "%Y-%m-%d"


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rolling 12m training pipeline: reuse monthly snapshots "
        "-> merge+dedup -> dataset -> train -> publish"
    )
    parser.add_argument(
        "--as-of",
        type=_parse_date,
        default=None,
        help=f"As-of date in {DATE_FORMAT} (default: today). Window is [as_of-12m, as_of).",
    )
    parser.add_argument(
        "--force-fetch",
        action="store_true",
        help="Force re-fetch monthly snapshots even if they exist.",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        default=True,
        help="Publish/update latest after training (default: enabled).",
    )
    parser.add_argument(
        "--months", type=int, default=12, help="Number of months in rolling window (default: 12)"
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    as_of = args.as_of.isoformat() if args.as_of else None

    res = trigger_rolling_12m.delay(
        as_of=as_of,
        force_fetch=bool(args.force_fetch),
        publish=bool(args.publish),
        months=int(args.months),
    )
    print("OK: rolling_12m task queued")
    print(f"- task_id: {res.id}")


if __name__ == "__main__":
    main()
