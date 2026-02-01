import time
from collections.abc import Callable

from fastapi import Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware

REGISTRY = CollectorRegistry()

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
    registry=REGISTRY,
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
    registry=REGISTRY,
)

PREDICTIONS_TOTAL = Counter(
    "model_predictions_total",
    "Total number of predictions served",
    ["realestate_type"],
    registry=REGISTRY,
)

PREDICTION_LATENCY = Histogram(
    "model_prediction_latency_seconds",
    "Time spent in model prediction logic",
    registry=REGISTRY,
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0),
)

TRAINING_STEP_DURATION_SECONDS = Histogram(
    "training_step_duration_seconds",
    "Duration of training steps in seconds",
    ["step"],
    registry=REGISTRY,
    buckets=(0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600, 1200, 2400),
)

ROLLING_SNAPSHOT_ROWS = Gauge(
    "rolling_snapshot_rows",
    "Row counts for rolling snapshots",
    ["metric"],
    registry=REGISTRY,
)

TRAINING_GATING_REASONS_TOTAL = Counter(
    "training_gating_reasons_total",
    "Number of gating results by reason",
    ["reason", "status"],
    registry=REGISTRY,
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        start = time.perf_counter()
        response: Response = await call_next(request)
        elapsed = time.perf_counter() - start

        path = request.url.path
        method = request.method
        status = str(response.status_code)

        HTTP_REQUESTS_TOTAL.labels(method=method, path=path, status=status).inc()
        HTTP_REQUEST_DURATION_SECONDS.labels(method=method, path=path).observe(elapsed)
        return response


def prometheus_response() -> Response:
    return Response(
        content=generate_latest(REGISTRY),
        media_type=CONTENT_TYPE_LATEST,
    )
