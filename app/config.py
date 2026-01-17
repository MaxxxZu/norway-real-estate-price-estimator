from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Runtime
    env: str = Field(default="local", alias="ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # HTTP binding
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")

    # CORS
    cors_origins: List[str] = Field(
        default_factory=lambda: ["*"],
        description="Allowed origins",
    )

    # Celery / RabbitMQ
    celery_broker_url: str = Field(
        default="amqp://guest:guest@rabbitmq:5672//",
        alias="CELERY_BROKER_URL",
    )
    celery_result_backend: str = Field(
        default="rpc://",
        alias="CELERY_RESULT_BACKEND",
    )

    # S3 / MinIO
    s3_endpoint: str = Field(default="http://minio:9000", alias="S3_ENDPOINT")
    s3_access_key: str = Field(default="minioadmin", alias="S3_ACCESS_KEY")
    s3_secret_key: str = Field(default="minioadmin", alias="S3_SECRET_KEY")
    s3_region: str = Field(default="eu-north-1", alias="S3_REGION")
    s3_secure: bool = Field(default=False, alias="S3_SECURE")

    s3_bucket_models: str = Field(default="ree-models", alias="S3_BUCKET_MODELS")
    s3_bucket_snapshots: str = Field(default="ree-snapshots", alias="S3_BUCKET_SNAPSHOTS")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="REE_",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
        populate_by_name=True,
    )


settings = Settings()
