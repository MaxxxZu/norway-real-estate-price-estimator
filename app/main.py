from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.ml.registry import ModelRegistry
from app.observability.logging import configure_logging, log
from app.observability.prometheus import PrometheusMiddleware
from app.observability.request_id import RequestIdMiddleware
from app.routes import router
from app.storage.s3 import S3Storage

configure_logging()
log().info("logging_configured", env=settings.env)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize and cleanup application resources."""
    # Startup
    log().info("initializing_resources")
    app.state.storage = S3Storage()
    app.state.registry = ModelRegistry(
        app.state.storage,
        refresh_seconds=settings.model_registry_refresh_seconds,
    )
    log().info("resources_initialized")

    yield

    # Shutdown
    log().info("shutting_down")


def create_app() -> FastAPI:
    api = FastAPI(
        title="Norway Real Estate Price Estimator",
        version="1.0.0",
        lifespan=lifespan,
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
