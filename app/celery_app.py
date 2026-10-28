from urllib.parse import urlsplit

from celery import Celery, Task, bootsteps
from kombu import Queue

from app.config import settings

celery_app = Celery(
    "ree",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend or None,
    include=["app.tasks.rolling"],
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
    task_ignore_result=False,  # Chord header tasks must persist results, including with overrides.
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


def validate_rolling_backend(app: Celery) -> None:
    """Check effective Celery configuration without instantiating/connecting a backend."""
    try:
        scheme = urlsplit(app.conf.result_backend or "").scheme.lower()
    except ValueError:
        scheme = ""
    if scheme not in {"redis", "rediss", "redis+socket"}:
        raise ValueError(
            "Rolling chords require a Redis result backend; configure "
            "CELERY_RESULT_BACKEND (takes precedence) or REE_CELERY_RESULT_BACKEND "
            "with redis://, rediss:// or redis+socket://."
        )
    if app.conf.task_ignore_result:
        raise ValueError("Rolling chords require task_ignore_result=False.")


class ValidateRollingBackend(bootsteps.StartStopStep):
    """Fail before consuming training tasks; inference imports remain independent."""

    def __init__(self, worker, **kwargs):
        if "training" in worker.app.amqp.queues.consume_from:
            validate_rolling_backend(worker.app)


celery_app.steps["worker"].add(ValidateRollingBackend)
