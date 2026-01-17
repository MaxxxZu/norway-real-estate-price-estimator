from app.worker.celery_app import celery_app


@celery_app.task(name="ree.ping")
def ping() -> dict:
    return {"status": "ok"}
