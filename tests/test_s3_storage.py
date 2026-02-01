from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from app.storage.s3 import S3Storage, S3StorageError


@pytest.fixture
def mock_boto_client():
    with patch("boto3.client") as mock:
        yield mock


def test_s3_storage_exists_true(mock_boto_client):
    """Test that exists returns True when head_object succeeds."""
    mock_s3 = mock_boto_client.return_value
    storage = S3Storage()

    assert storage.exists("bucket", "key") is True
    mock_s3.head_object.assert_called_once_with(Bucket="bucket", Key="key")


def test_s3_storage_exists_false(mock_boto_client):
    """Test that exists returns False when head_object raises 404."""
    mock_s3 = mock_boto_client.return_value
    error_response = {"Error": {"Code": "404", "Message": "Not Found"}}
    mock_s3.head_object.side_effect = ClientError(error_response, "HeadObject")

    storage = S3Storage()
    assert storage.exists("bucket", "key") is False


def test_s3_storage_exists_error(mock_boto_client):
    """Test that exists raises S3StorageError for other errors."""
    mock_s3 = mock_boto_client.return_value
    error_response = {"Error": {"Code": "500", "Message": "Internal Error"}}
    mock_s3.head_object.side_effect = ClientError(error_response, "HeadObject")

    storage = S3Storage()
    with pytest.raises(S3StorageError):
        storage.exists("bucket", "key")


def test_get_bytes_success(mock_boto_client):
    """Test successful retrieval of bytes."""
    mock_s3 = mock_boto_client.return_value
    mock_body = MagicMock()
    mock_body.read.return_value = b"content"
    mock_s3.get_object.return_value = {"Body": mock_body}

    storage = S3Storage()
    result = storage.get_bytes("bucket", "key")
    assert result == b"content"


def test_get_bytes_failure(mock_boto_client):
    """Test failure during get_bytes."""
    mock_s3 = mock_boto_client.return_value
    mock_s3.get_object.side_effect = ClientError({}, "GetObject")

    storage = S3Storage()
    with pytest.raises(S3StorageError):
        storage.get_bytes("bucket", "key")


def test_put_bytes_success(mock_boto_client):
    """Test successful put_bytes."""
    mock_s3 = mock_boto_client.return_value

    storage = S3Storage()
    storage.put_bytes("bucket", "key", b"data", "text/plain")

    mock_s3.put_object.assert_called_once_with(
        Bucket="bucket", Key="key", Body=b"data", ContentType="text/plain"
    )


def test_get_json_success(mock_boto_client):
    """Test successful JSON retrieval."""
    mock_s3 = mock_boto_client.return_value
    mock_body = MagicMock()
    mock_body.read.return_value = b'{"foo": "bar"}'
    mock_s3.get_object.return_value = {"Body": mock_body}

    storage = S3Storage()
    data = storage.get_json("bucket", "key")
    assert data == {"foo": "bar"}


def test_get_json_invalid(mock_boto_client):
    """Test invalid JSON raises error."""
    mock_s3 = mock_boto_client.return_value
    mock_body = MagicMock()
    mock_body.read.return_value = b"invalid-json"
    mock_s3.get_object.return_value = {"Body": mock_body}

    storage = S3Storage()
    with pytest.raises(S3StorageError, match="Invalid JSON"):
        storage.get_json("bucket", "key")
