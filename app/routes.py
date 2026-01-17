from fastapi import APIRouter

from app.config import settings

router = APIRouter()


@router.get("/healthz", tags=["health"])
def healthz() -> dict:
    return {
        "status": "ok",
        "service": "ree",
        "env": settings.env,
    }
