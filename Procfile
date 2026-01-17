api: sh -lc "set -a && . ./.env && set +a && uv run uvicorn app.main:app --host ${REE_HOST:-0.0.0.0} --port ${REE_PORT:-8000} --reload"
worker: sh -lc "set -a && . ./.env && set +a && uv run watchfiles --filter python \"celery -A app.celery_app:celery_app worker -l ${REE_LOG_LEVEL:-INFO}\" app"
beat: sh -lc "set -a && . ./.env && set +a && uv run watchfiles --filter python \"celery -A app.celery_app:celery_app beat -l ${REE_LOG_LEVEL:-INFO}\" app"
