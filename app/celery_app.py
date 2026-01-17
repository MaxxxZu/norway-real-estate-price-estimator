from celery import Celery
from celery.schedules import crontab

from app.config import settings

celery_app = Celery(
    "ree",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend or None,
)

celery_app.conf.update(
    timezone=settings.celery_timezone,
    enable_utc=True,
    task_track_started=True,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
)

celery_app.autodiscover_tasks(["app.tasks"])

celery_app.conf.beat_schedule = {
    "retrain-previous-month": {
        "task": "app.tasks.retrain.retrain_previous_month",
        "schedule": crontab(minute=5, hour=3, day_of_month="1"),
    }
}
