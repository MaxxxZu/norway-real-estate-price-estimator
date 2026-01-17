import json
from dataclasses import dataclass
from typing import Any, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from app.config import settings


class S3StorageError(RuntimeError):
    pass


@dataclass(frozen=True)
class S3ObjectRef:
    bucket: str
    key: str


class S3Storage:
    def __init__(self):
        self._client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint or None,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
            region_name=settings.s3_region,
            use_ssl=settings.s3_secure,
        )

    def get_bytes(self, bucket: str, key: str) -> bytes:
        try:
            resp = self._client.get_object(Bucket=bucket, Key=key)
            body = resp["Body"].read()
            return body
        except (ClientError, BotoCoreError) as e:
            raise S3StorageError(f"Failed to get s3://{bucket}/{key}") from e

    def put_bytes(
        self,
        bucket: str,
        key: str,
        data: bytes,
        content_type: Optional[str] = None,
    ) -> None:
        try:
            extra = {}
            if content_type:
                extra["ContentType"] = content_type
            self._client.put_object(Bucket=bucket, Key=key, Body=data, **extra)
        except (ClientError, BotoCoreError) as e:
            raise S3StorageError(f"Failed to put s3://{bucket}/{key}") from e

    def get_json(self, bucket: str, key: str) -> dict[str, Any]:
        raw = self.get_bytes(bucket=bucket, key=key)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception as e:
            raise S3StorageError(f"Invalid JSON at s3://{bucket}/{key}") from e

    def put_json(self, bucket: str, key: str, obj: dict[str, Any]) -> None:
        data = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
        self.put_bytes(bucket=bucket, key=key, data=data, content_type="application/json")
