# syntax=docker/dockerfile:1.7

FROM ghcr.io/astral-sh/uv:0.4.20 AS uv
FROM python:3.11-slim-bookworm AS base

RUN apt-get update \
 && apt-get -y upgrade --no-install-recommends \
 && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /bin/uv

ENV UV_SYSTEM_PYTHON=1 \
    UV_LINK_MODE=copy \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN useradd -m -U appuser
WORKDIR /app

COPY --chown=appuser:appuser pyproject.toml uv.lock ./
RUN /bin/uv sync --frozen || /bin/uv sync

COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser scripts ./scripts

ENV PYTHONPATH=/app

USER appuser

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
