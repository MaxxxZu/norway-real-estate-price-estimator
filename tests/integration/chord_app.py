"""Synthetic tasks for an actual worker; never call data API, fit or storage."""

from app.celery_app import celery_app


@celery_app.task(name="tests.chord.echo")
def echo(value):
    return value


@celery_app.task(name="tests.chord.total")
def total(values):
    return {"values": sorted(values), "total": sum(values)}
