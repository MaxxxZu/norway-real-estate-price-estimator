import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from structlog.contextvars import bind_contextvars, clear_contextvars

from app.observability.logging import log


class RequestIdMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, header_name: str = "X-Request-Id"):
        super().__init__(app)
        self.header_name = header_name

    async def dispatch(self, request: Request, call_next: Callable):
        clear_contextvars()

        request_id = request.headers.get(self.header_name) or str(uuid.uuid4())
        bind_contextvars(request_id=request_id)

        start = time.perf_counter()
        try:
            response: Response = await call_next(request)
        except Exception:
            log().exception(
                "http_request_failed",
                method=request.method,
                path=request.url.path,
            )
            raise

        latency_ms = int((time.perf_counter() - start) * 1000)
        response.headers[self.header_name] = request_id

        log().info(
            "http_request_completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            latency_ms=latency_ms,
        )
        return response
