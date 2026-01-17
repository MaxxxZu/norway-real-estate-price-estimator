from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routes import router


def create_app() -> FastAPI:
    api = FastAPI(
        title="Norway Real Estate Price Estimator",
        version="0.1.0",
    )

    api.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    api.include_router(router)

    return api


app = create_app()
