from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.observability.logging import configure_logging, log
from app.observability.prometheus import PrometheusMiddleware
from app.observability.request_id import RequestIdMiddleware
from app.routes import router

configure_logging()
log().info("logging_configured", env=settings.env)


def create_app() -> FastAPI:
    api = FastAPI(
        title="Norway Real Estate Price Estimator",
        version="1.0.0",
    )

    api.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    api.add_middleware(RequestIdMiddleware)
    api.add_middleware(PrometheusMiddleware)
    api.include_router(router)

    return api


app = create_app()
