from datetime import datetime, timezone

from app.config import settings
from app.storage.s3 import S3Storage


def main() -> None:
    storage = S3Storage()

    model_version = "stub-v1"
    artifact_key = f"models/{model_version}/model.json"

    artifact = {
        "type": "stub",
        "params": {
            "usable_area_coef": 50_000,
            "total_area_coef": 5_000,
        },
    }

    latest = {
        "model_version": model_version,
        "type": "stub",
        "artifact_key": artifact_key,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    storage.put_json(bucket=settings.s3_bucket_models, key=artifact_key, obj=artifact)
    storage.put_json(bucket=settings.s3_bucket_models, key="latest.json", obj=latest)

    print(f"OK: uploaded {artifact_key} and latest.json to bucket {settings.s3_bucket_models}")


if __name__ == "__main__":
    main()
