from celery import Celery, Task
from kombu import Queue

from app.config import settings

celery_app = Celery(
    "ree",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend or None,
)


class TrainingTask(Task):
    autoretry_for = (Exception,)
    retry_backoff = 60
    retry_jitter = True
    retry_kwargs = {"max_retries": 5}
    soft_time_limit = 900
    time_limit = 1200


class TrainingLongTask(TrainingTask):
    soft_time_limit = 1800
    time_limit = 2400


class TrainingShortTask(TrainingTask):
    soft_time_limit = 120
    time_limit = 240

celery_app.conf.update(
    timezone=settings.celery_timezone,
    enable_utc=True,
    task_track_started=True,
    task_ignore_result=not bool(settings.celery_result_backend),
    task_default_retry_delay=60,
    task_max_retries=5,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    # RabbitMQ reliability
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
    task_queues=(
        Queue("celery"),
        Queue("training"),
    ),
)

celery_app.autodiscover_tasks(["app.tasks"])
