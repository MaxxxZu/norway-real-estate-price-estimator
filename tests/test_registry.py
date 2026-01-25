import time
from unittest.mock import Mock, patch

import pytest

from app.ml.registry import ModelRegistry, ModelNotReadyError
from app.ml.stub import StubPredictor
from app.storage.s3 import S3Storage, S3StorageError


@pytest.fixture
def mock_storage():
    return Mock(spec=S3Storage)


def test_get_predictor_cached(mock_storage):
    """Test that predictor is returned from cache if fresh."""
    registry = ModelRegistry(mock_storage, refresh_seconds=60)
    
    # Manually seed cache
    mock_predictor = Mock(spec=StubPredictor)
    registry._cached_predictor = mock_predictor
    registry._cached_at = time.time()
    
    assert registry.get_predictor() is mock_predictor
    mock_storage.get_json.assert_not_called()


def test_get_predictor_refreshes_cache(mock_storage):
    """Test that predictor is refreshed when cache expires."""
    registry = ModelRegistry(mock_storage, refresh_seconds=60)
    
    # Mock S3 responses
    mock_storage.get_json.side_effect = [
        # latest.json
        {
            "model_version": "v1",
            "type": "stub",
            "artifact_key": "models/v1/artifact.json"
        },
        # artifact.json
        {"params": {}}
    ]
    
    predictor = registry.get_predictor()
    
    assert isinstance(predictor, StubPredictor)
    assert registry._cached_version == "v1"
    assert mock_storage.get_json.call_count == 2


def test_load_latest_missing(mock_storage):
    """Test error when latest.json is missing."""
    registry = ModelRegistry(mock_storage)
    mock_storage.get_json.side_effect = S3StorageError("Not found")
    
    with pytest.raises(ModelNotReadyError, match="Model registry is not initialized"):
        registry.get_predictor()


def test_load_latest_invalid_content(mock_storage):
    """Test error when latest.json is invalid."""
    registry = ModelRegistry(mock_storage)
    mock_storage.get_json.return_value = {}  # Empty dict
    
    with pytest.raises(ModelNotReadyError, match="missing required fields"):
        registry.get_predictor()


def test_build_predictor_unsupported_type(mock_storage):
    """Test error for unsupported model type."""
    registry = ModelRegistry(mock_storage)
    mock_storage.get_json.return_value = {
        "model_version": "v1",
        "type": "tensorflow",  # Unsupported
        "artifact_key": "key"
    }
    
    with pytest.raises(ModelNotReadyError, match="Unsupported model type"):
        registry.get_predictor()


def test_build_predictor_sklearn(mock_storage):
    """Test building sklearn predictor."""
    registry = ModelRegistry(mock_storage)
    
    mock_storage.get_json.side_effect = [
        # latest.json
        {
            "model_version": "v2",
            "type": "sklearn",
            "artifact_key": "models/v2/model.joblib"
        },
        # feature_schema.json
        {"prediction_transform": "log1p"}
    ]
    mock_storage.get_bytes.return_value = b"fake-joblib-data"
    
    # patch joblib.load to avoid needing real data
    with patch("joblib.load", return_value="mock-model"):
        predictor = registry.get_predictor()
        
    assert predictor.model_version == "v2"
