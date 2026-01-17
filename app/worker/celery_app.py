from celery import Celery

from app.config import settings

celery_app = Celery(
    "ree",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.autodiscover_tasks(["app.worker.tasks"])
