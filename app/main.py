from fastapi import FastAPI

from app.routes import router


def create_app() -> FastAPI:
    api = FastAPI(
        title="Norway Real Estate Price Estimator",
        version="0.1.0",
    )
    api.include_router(router)
    return api


app = create_app()
