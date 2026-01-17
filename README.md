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

## Local development
### 0) Prerequisites
- Docker (for RabbitMQ + MinIO)
- uv (dependency management)
- (optional) GitHub Actions CI runs ruff + pytest

### 1) Configure environment
```bash
cp .env.example .env
uv lock
make compose-up
```
### 2) Start infrastructure (RabbitMQ + MinIO)
```bash
make compose-up
```
## Useful Links

| Service | URL | Credentials |
|---------|-----|-------------|
| **RabbitMQ Management UI** | http://localhost:15672 | `guest` / `guest` |
| **MinIO Console** | http://localhost:9001 | See `.env` |
| **MinIO S3 Endpoint** | http://localhost:9000 | - |

### 3) Start application (FastAPI + Celery worker)
```bash
honcho start
```

## Open:
- [Swagger](http://localhost:8080/docs)
- [ReDoc](http://localhost:8080/redoc)
- [Health](http://localhost:8080/healthz)

## 📝 Notes

### Development Architecture
In the development environment:
- ✅ Application code runs **natively on your host** (fast hot-reload)
- 🐳 RabbitMQ and MinIO run in **Docker containers**

### Storage Strategy
- **Local Development:** MinIO serves as an S3-compatible storage solution
- **Production:** Same codebase connects to AWS S3 or any S3-compatible provider
- **Configuration:** Switch between environments using environment variables only—no code changes required
