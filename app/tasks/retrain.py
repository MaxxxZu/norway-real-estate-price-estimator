from datetime import date, timedelta

from celery import shared_task

from app.training.pipeline import run_training_pipeline


def _previous_month_range(today: date) -> tuple[date, date]:
    first_this_month = today.replace(day=1)
    last_prev_month = first_this_month - timedelta(days=1)
    first_prev_month = last_prev_month.replace(day=1)
    return first_prev_month, last_prev_month


@shared_task(name="app.tasks.retrain.retrain_previous_month")
def retrain_previous_month() -> dict:
    start, end = _previous_month_range(date.today())
    return run_training_pipeline(
        start_date=start,
        end_date=end,
        dry_run=False,
        train=True,
        publish=True,
    )


@shared_task(name="app.tasks.retrain.retrain_range")
def retrain_range(start_date: str, end_date: str, publish: bool = True) -> dict:
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    return run_training_pipeline(
        start_date=start,
        end_date=end,
        dry_run=False,
        train=True,
        publish=bool(publish),
    )
