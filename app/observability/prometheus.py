import time
from typing import Callable

from fastapi import Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
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
