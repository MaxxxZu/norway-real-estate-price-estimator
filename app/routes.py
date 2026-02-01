import socket
from typing import Any
from urllib.parse import urlparse

from fastapi import APIRouter, Body, Depends, HTTPException, status

from app.api.examples import ESTIMATE_REQUEST_EXAMPLES
from app.api.routes.metrics import router as metrics_router
from app.celery_app import celery_app
from app.config import settings
from app.dependencies import get_predictor
from app.ml.base import Predictor
from app.schemas import EstimateRequest, EstimateResponse, HealthCheckResponse
from app.services.estimate_service import estimate_batch

router = APIRouter()
router.include_router(metrics_router)


@router.get(
    "/health",
    tags=["healthcheck"],
    summary="Perform a Health Check",
    response_description="Return HTTP Status Code 200 (OK)",
    status_code=status.HTTP_200_OK,
    response_model=HealthCheckResponse,
)
async def get_health() -> HealthCheckResponse:
    return HealthCheckResponse(status="ok", env=settings.env)


def _tcp_check(url: str) -> dict[str, Any]:
    parsed = urlparse(url)
    host = parsed.hostname
    if not host:
        return {"ok": False, "error": "missing_host"}

    port = parsed.port
    if port is None:
        if parsed.scheme in {"amqps"}:
            port = 5671
        elif parsed.scheme in {"amqp"}:
            port = 5672
        elif parsed.scheme in {"rediss"}:
            port = 6380
        elif parsed.scheme in {"redis"}:
            port = 6379
        else:
            return {"ok": False, "error": "unknown_scheme"}

    try:
        with socket.create_connection((host, port), timeout=1):
            return {"ok": True, "host": host, "port": port}
    except OSError as e:
        return {"ok": False, "host": host, "port": port, "error": str(e)}


@router.post(
    "/estimate",
    response_model=EstimateResponse,
    responses={
        503: {"description": "Model not ready"},
    },
    tags=["estimation"],
)
def estimate(
    payload: EstimateRequest = Body(..., openapi_examples=ESTIMATE_REQUEST_EXAMPLES),
    predictor: Predictor = Depends(get_predictor),
) -> Any:
    if not payload:
        raise HTTPException(status_code=422, detail="Empty payload")

    return estimate_batch(payload, predictor=predictor)


@router.get(
    "/ready",
    tags=["readycheck"],
    summary="Perform a Ready Check",
    response_description="Return HTTP Status Code 200 (OK)",
    status_code=status.HTTP_200_OK,
)
def get_ready(predictor: Predictor = Depends(get_predictor)) -> dict[str, str]:
    return {"status": "ready"}


@router.get(
    "/health/queue",
    tags=["healthcheck"],
    summary="Check broker/backend connectivity and worker liveness",
    status_code=status.HTTP_200_OK,
)
def get_queue_health() -> dict[str, Any]:
    broker = _tcp_check(settings.celery_broker_url)

    backend_url = settings.celery_result_backend or ""
    if backend_url.startswith(("redis://", "rediss://")):
        backend = _tcp_check(backend_url)
    else:
        backend = {"ok": True, "skipped": True}

    workers: list[str] = []
    try:
        replies = celery_app.control.ping(timeout=1) or []
        for reply in replies:
            if isinstance(reply, dict):
                workers.extend(reply.keys())
        workers_ok = bool(workers)
        workers_info: dict[str, Any] = {
            "ok": workers_ok,
            "count": len(workers),
            "workers": workers,
        }
    except Exception as e:
        workers_info = {"ok": False, "error": str(e), "count": 0, "workers": []}

    status_value = "ok"
    if not broker.get("ok"):
        status_value = "error"
    elif not workers_info.get("ok"):
        status_value = "degraded"
    elif not backend.get("ok"):
        status_value = "degraded"

    return {
        "status": status_value,
        "broker": broker,
        "backend": backend,
        "workers": workers_info,
    }
