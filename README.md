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

### 3) Create stub model
```bash
uv run python -m scripts.bootstrap_model_registry
```

### 4) Start application (FastAPI + Celery worker)
```bash
honcho start
```

## Open:
- [Swagger](http://localhost:8080/docs)
- [ReDoc](http://localhost:8080/redoc)
- [Health](http://localhost:8080/health)

## Training (dry-run snapshots)
Fetch data for a date range, build dataset, and upload snapshots to MinIO:

```bash
uv run python -m scripts.train --start-date 2026-01-01 --end-date 2026-01-31 --dry-run
```

## Retraining (scheduled)
Local dev runs Celery worker + Celery beat via Procfile.

- Monthly retrain runs on the 1st day at 03:05 (Europe/Ljubljana) and trains the previous month, then publishes to MinIO and updates `ree-models/latest.json`.

Manual retrain:
```bash
uv run celery -A app.celery_app.celery_app call app.tasks.retrain.retrain_range --args='["2026-01-01","2026-01-31", true]'
```

# Deploy to kind
## Prerequisites

- `kubectl`
- `kind`
- `helm`
- `docker`
- `k9s` (optional)
---
## Create cluster
```bash
kind create cluster --name ree
kubectl cluster-info --context kind-ree
```
---
## Build and load image to kind
```bash
TAG=dev-$(git rev-parse --short HEAD)
docker build -t ree:$TAG -f deploy/Dockerfile .
kind load docker-image ree:$TAG --name ree
```
---
## Create namespace + secret from .env.kind
```bash
kubectl create namespace ree 2>/dev/null || true
kubectl -n ree delete secret ree-secrets --ignore-not-found
kubectl -n ree create secret generic ree-secrets --from-env-file=.env.kind
```
---
## Install with Helm (kind values)
```bash
helm upgrade --install ree deploy/helm -n ree -f deploy/helm/values_kind.yaml
```
---
## Port-forward
```bash
kubectl -n ree port-forward svc/ree-api 8000:80
kubectl -n ree port-forward svc/ree-minio 9001:9001
```
---
## Run training once (manual, first time)
```bash
kubectl -n ree create job --from=cronjob/ree-training ree-training-manual-1
```
---
## Enable CronJob in kind (optional)
Set `training.suspend: false` in `values.yaml` and run:
```bash
helm upgrade --install ree deploy/helm -n ree -f deploy/helm/values_kind.yaml
```
---
### Note:
- Use you k8s config, example: `KUBECONFIG=~/.kube/kind-config kubectl`



## 📝 Notes

### Development Architecture
In the development environment:
- ✅ Application code runs **natively on your host** (fast hot-reload)
- 🐳 RabbitMQ and MinIO run in **Docker containers**

### Storage Strategy
- **Local Development:** MinIO serves as an S3-compatible storage solution
- **Production:** Same codebase connects to AWS S3 or any S3-compatible provider
- **Configuration:** Switch between environments using environment variables only—no code changes required

## 🔄 Pipeline results

**Period:** 01.01.2026 - 31.01.2026

📄 `manifest.json`
```json
{
  "period": {
    "start_date": "2026-01-01",
    "end_date": "2026-01-31"
  },
  "counts": {
    "turnovers_raw": 6559,
    "turnovers_normalized": 6179,
    "cadastral_unit_ids": 6179,
    "properties_matched": 2900,
    "rows_raw": 2459,
    "rows_trainable": 2219
  },
  "dropped_reasons": {
    "invalid:total_area": 191,
    "invalid:total_area_lt_bra": 49
  },
  "dry_run": true
}
```

## Sklearn model Evaluation Metrics

### Overall Performance

- **MAE:** 963,341
- **RMSE:** 1,610,995

> Average absolute prediction error is ~963k.
> Higher RMSE indicates the presence of large outliers.

---

### MAE by Real Estate Type

| Type | MAE |
|------|----:|
| leilighet | 799,366 |
| hytte | 928,496 |
| rekkehus | 970,871 |
| tomannsbolig | 1,115,249 |
| enebolig | 1,438,171 |

---

### MAE by Municipality (Top)

| Municipality | MAE |
|-------------:|----:|
| 3205 | 448,371 |
| 3107 | 751,234 |
| 4204 | 821,650 |
| 1508 | 918,516 |
| 5001 | 937,269 |
| 4601 | 984,198 |
| 3301 | 966,644 |
| 3201 | 1,065,213 |
| 1108 | 1,306,385 |
| 301 | 1,432,622 |

---

### Dataset Size

- **Train:** 1,783 samples
- **Test:** 446 samples

---

### Notes

- Best performance on apartments (*leilighet*)
- Higher errors for detached houses (*enebolig*)
- Error varies significantly across municipalities
- Dataset size is relatively small, contributing to higher variance