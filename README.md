# Norway Real Estate Price Estimator

End-to-end AI system for estimating **current market prices** of residential real estate in Norway.

## Key features
- Current price estimation via FastAPI
- Batch `/estimate` API
- Monthly model retraining on real transaction data
- Async training pipeline (Celery + RabbitMQ)
- S3-compatible storage for data snapshots and models (MinIO)
- Kubernetes-ready deployment (Helm)

## Tech stack
- FastAPI, Uvicorn
- `uv` (dependency manager) with `pyproject.toml` + `uv.lock`
- Celery + RabbitMQ
- MinIO (S3-compatible)
- Docker & Docker Compose
- Kubernetes (Helm)
- scikit-learn (v1), extensible to LightGBM

## Project status
🚧 Work in progress.
This repository is being built step-by-step as a production-like AI system.
