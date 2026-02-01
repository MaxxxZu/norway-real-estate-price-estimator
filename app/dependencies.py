"""Dependency injection utilities for FastAPI."""

from fastapi import HTTPException, Request

from app.ml.base import Predictor
from app.ml.registry import ModelNotReadyError, ModelRegistry


def get_registry(request: Request) -> ModelRegistry:
    """Get ModelRegistry from app.state."""
    return request.app.state.registry


def get_predictor(request: Request) -> Predictor:
    """Get current Predictor from ModelRegistry."""
    registry: ModelRegistry = request.app.state.registry
    try:
        return registry.get_predictor()
    except ModelNotReadyError as e:
        raise HTTPException(
            status_code=503, detail={"message": "model_not_ready", "reason": str(e)}
        ) from e
