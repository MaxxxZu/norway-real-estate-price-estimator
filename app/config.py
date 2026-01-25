from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    env: str = Field(default="local")
    log_level: str = Field(default="INFO")

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)

    cors_origins: List[str] = Field(
        default_factory=lambda: ["*"],
        description="Allowed origins",
    )

    celery_broker_url: str = Field(
        default="amqp://guest:guest@rabbitmq:5672//",
    )
    celery_result_backend: str = Field(
        default="rpc://",
    )
    celery_timezone: str = Field(default="Europe/Ljubljana")

    api_base_url: str = Field(default="http://127.0.0.1:9080")
    api_key: str = Field(default="")

    s3_endpoint: str = Field(default="http://minio:9000")
    s3_access_key: str = Field(default="minioadmin")
    s3_secret_key: str = Field(default="minioadmin")
    s3_region: str = Field(default="eu-north-1")
    s3_secure: bool = Field(default=False)

    s3_bucket_models: str = Field(default="ree-models")
    s3_bucket_snapshots: str = Field(default="ree-snapshots")

    model_registry_refresh_seconds: int = Field(default=60)

    # Training publish gating
    train_min_rows: int = Field(default=500)

    gate_overall_mae_max_degrade_pct: float = Field(default=0.05)
    gate_enebolig_mae_max_degrade_pct: float = Field(default=0.07)
    gate_overall_wape_max_degrade_pct: float = Field(default=0.10)
    gate_enebolig_wape_max_degrade_pct: float = Field(default=0.12)

    # Metrics summary thresholds (UI/Report)
    metrics_mdape_good_threshold: float = Field(default=0.12)
    metrics_mdape_ok_threshold: float = Field(default=0.18)
    metrics_ae_p90_tail_risk_nok: int = Field(default=2_000_000)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="REE_",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


settings = Settings()
