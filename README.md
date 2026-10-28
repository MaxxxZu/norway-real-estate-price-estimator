# Norway Real Estate Price Estimator

[![Test](https://github.com/MaxxxZu/norway-real-estate-price-estimator/actions/workflows/ci.yml/badge.svg)](https://github.com/MaxxxZu/norway-real-estate-price-estimator/actions/workflows/ci.yml)
[![Release](https://github.com/MaxxxZu/norway-real-estate-price-estimator/actions/workflows/docker-build.yml/badge.svg)](https://github.com/MaxxxZu/norway-real-estate-price-estimator/actions/workflows/docker-build.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**Production-grade AI / MLOps project** for estimating **current market prices** of residential real estate in Norway based on historical transaction data.

The project demonstrates a **full end-to-end AI system**:

> data ingestion → feature engineering → model training → quality gating → model registry → inference API → Kubernetes → CI/CD

This repository is intentionally built as a **realistic production-like system**, not a toy ML notebook.

---

## 🚀 What this project does

- Fetches real real-estate transactions from an external API
- Builds validated, reproducible training datasets
- Trains ML models for price estimation
- Computes rich quality metrics (absolute & relative)
- Applies **quality gating** before publishing models
- Stores immutable artifacts in object storage (S3 compatible)
- Serves predictions via FastAPI
- Retrains models automatically on a schedule (Kubernetes CronJob, rolling 12-month window)
- Deploys to Kubernetes via Helm (kind / GKE)
- Supports full CI/CD via GitHub Actions

---

## 🧱 High-level architecture
```mermaid
flowchart LR
    subgraph External
        EXT_API[External Real Estate API]
    end

    subgraph Training["Training (Kubernetes CronJob)"]
        FETCH[Fetch monthly data]
        MERGE[Merge & deduplicate<br/>rolling 12m]
        DATASET[Dataset + Manifest]
        TRAIN[Train Model]
        EVAL[Compute metrics]
        GATE[Quality gating]
        PUBLISH[Publish model]
    end

    subgraph Storage["Object Storage (S3 compatible)"]
        SNAP[Snapshots]
        REG[Model registry + models]
    end

    subgraph API["Inference (FastAPI)"]
        READY[/ready/]
        EST[/estimate/]
        METRICS[/metrics/]
        PROM[/metrics/prometheus/]
        REGISTRY[ModelRegistry - cached]
        PRED[SklearnPredictor]
    end

    EXT_API --> FETCH
    CRON --> CHORD
    CHORD --> FETCH
    FETCH --> MERGE
    MERGE --> DATASET
    DATASET --> TRAIN
    TRAIN --> GATE
    GATE -->|pass| PUBLISH
    DATASET --> SNAPSHOTS
    PUBLISH --> MODELS

    MODELS --> REGISTRY
    REGISTRY --> PREDICT
    PREDICT --> FASTAPI
```

## 🔁 Inference request flow

```mermaid
sequenceDiagram
  participant Client
  participant API as FastAPI
  participant Reg as ModelRegistry
  participant S3 as Object Storage
  participant Pred as Predictor

  Client->>API: POST /estimate (batch)
  API->>Reg: get_predictor() (cached with TTL)
  alt cache miss
    Reg->>S3: GET latest.json
    Reg->>S3: GET models/<ver>/model.pkl
    Reg->>S3: GET models/<ver>/feature_schema.json
  end
  Reg-->>API: Predictor
  API->>Pred: predict(batch)
  Pred-->>API: prices
  API-->>Client: 200 OK (or 422 / 503)
```

## 📦 Core components

### 1️⃣ Training pipeline
Runs as a Kubernetes **CronJob** (and can be executed manually). Current mode is
**rolling 12-month** training with monthly snapshot reuse and de-duplication.

### Pipeline stages

- Fetch transactions for a given period
- Normalize and validate rows
- Build a reproducible training dataset
- Train a regression model
- Evaluate metrics
- Apply quality gating
- Publish artifacts if gating passes

### Artifacts stored per training run

- `rows_raw.jsonl` — fetched source rows (including rejected rows)
- `dataset.parquet` — trainable dataset
- `manifest.json` — versioned completion, fingerprints, source lineage and drop reasons
- `model.pkl` — trained model
- `metrics.json` — evaluation metrics
- `feature_schema.json` — versioned shared features and fixed reference year
- `latest.json` — pointer to the active model

---

Snapshot revision paths, normalization, feature contracts and the owner-selected
legacy rejection/transition policy are documented in [Data contracts](docs/DATA_CONTRACTS.md).

### 2️⃣ Model Registry

- Implemented on top of object storage
- Each model version is immutable
- `latest.json` points to the active production model
- API refuses to serve predictions until a valid model exists
- Registry is cached in-memory with TTL for performance

---

### 3️⃣ Inference API (FastAPI)

### Endpoints

- `GET /health` — liveness probe
- `GET /ready` — readiness probe (model loaded)
- `POST /estimate` — batch price estimation
- `GET /metrics` — raw model metrics
- `GET /metrics/summary` — human-friendly quality report
- `GET /metrics/prometheus` - Prometheus scrape

### Design principles

- Batch-first API (property_id → features)
- Strict validation (“better no estimate than a wrong one”)
- Invalid requests return HTTP 422 with FastAPI/Pydantic validation errors (list in `detail`)
- Predictor logic isolated from HTTP layer
- Explicit readiness based on model availability

---

### 4️⃣ Storage

* **Local / kind:** MinIO
* **GKE:** Google Cloud Storage via S3 interoperability
* Same code works in both environments

> ⚠️ **Note:** boto3 uploads require `ContentLength` to avoid aws-chunked uploads when using GCS S3 compatibility.
---

## 🧠 Machine Learning

### Current approach

- Model: `CatBoostRegressor`
- Target: `log1p(price)` with inverse transform on inference
- Handles categorical + numeric features natively
- Stronger performance vs sklearn baseline
- Sliding window: rolling 12-month window (monthly snapshots + merge)

### Feature groups

- Property size (`total_area`, `usable_area`)
- Location (`lat`, `lon`, `municipality`)
- Building metadata (`year built`, `floor`, `rooms`, `bedrooms`)
- Real estate type
- Derived ratios (e.g. area ratios)

### Metrics used
- **MAE** (NOK)
- **RMSE** (NOK)
- **MAPE / MdAPE** (relative error; robust vs outliers)
- **AE_p90** (tail risk: 90th percentile absolute error)
- **WAPE** (weighted absolute percentage error)

---


## 📊 Model Quality & Metrics

**GET `/metrics`** — raw metrics JSON from the model registry

**GET `/metrics/summary`** — human-friendly summary (focus on realestate_type segments)

### Includes

- Overall metrics (MAE, RMSE, MdAPE, WAPE, AE_p90)
- Metrics per `realestate_type`
- Best / worst performing segments
- Automatic risk flags
- Tail-risk detection
- Thresholds used for interpretation

### Example usage

```bash
curl -s http://localhost:8000/metrics/summary | jq
```
This endpoint intentionally reads like a model evaluation report, not a raw ML dump.

---

## 🚦 Quality Gating

A newly trained model is published **only if**:
- Minimum dataset size is met
- Overall quality does not degrade beyond configured thresholds
- Key segment (`enebolig`) does not regress excessively
- Optional relative error metrics (MdAPE / WAPE) stay within bounds

If gating fails → **previous model remains active**.

This prevents silent production regressions.

---

## 🔍 Observability

### Logs
- JSON structured logs (structlog)
- Each request includes `request_id` (from `X-Request-Id` or auto-generated)

### Prometheus metrics
- `GET /metrics/prometheus` exposes OpenMetrics format:
  - `http_requests_total`
  - `http_request_duration_seconds`

---

## ☸️ Kubernetes & Deployment

- Helm chart located in `deploy/helm`
- Environments
    - `kind` — local development
    - `GKE` — production-like cluster
- Explicit CPU / memory requests & limits
- Training runs as a `CronJob` that triggers rolling 12-month training
- API runs as a `Deployment` with readiness probes
- RabbitMQ and MinIO deployed as Helm releases

---

## 🔄 CI/CD

**Release flow:**

1. Push git tag `vX.Y.Z`
2. GitHub Actions:
   * build Docker image
   * push to GHCR
   * deploy to GKE via Helm
   * update Kubernetes secrets
3. Rolling update with readiness checks

## 🧪 Local development

### Prerequisites

- Docker
- Python 3.11+
- `uv`
- `kind`
- `kubectl`
- `helm`
- RabbitMQ (required for Celery tasks)
- MinIO (optional, or use S3-compatible storage)

### Run locally
```bash
cp .env.example .env
uv lock
make compose-up
honcho start
```
## Open:
- [Swagger](http://localhost:8080/docs)
- [ReDoc](http://localhost:8080/redoc)
- [Health](http://localhost:8080/health)


---

## 🚢 Deploy to kind
```bash
kind create cluster --name ree
kubectl cluster-info --context kind-ree

TAG=dev-$(git rev-parse --short HEAD)
docker build -t ree:$TAG -f deploy/Dockerfile .
kind load docker-image ree:$TAG --name ree

kubectl create namespace ree 2>/dev/null || true
kubectl -n ree delete secret ree-secrets --ignore-not-found
kubectl -n ree create secret generic ree-secrets --from-env-file=.env.kind

helm upgrade --install ree deploy/helm -n ree -f deploy/helm/values_kind.yaml

kubectl -n ree port-forward svc/ree-api 8000:80
kubectl -n ree port-forward svc/ree-minio 9001:9001
```

### Run initial training manually:
```bash
kubectl -n ree create job --from=cronjob/ree-training ree-training-manual
```
### Trigger rolling training manually (outside CronJob):
```bash
uv run python -m scripts.rolling_12m --publish
```
### Note:
- Use you k8s config, example: `KUBECONFIG=~/.kube/kind-config kubectl`

---

## 🛣️ Roadmap (next steps)
- Model v4 (feature engineering + calibration)
- Richer geo features
- Drift detection
- Shadow evaluation
- Online / offline metric comparison

---

## 📝 Notes
This project is intentionally built as a production-like system, not just a notebook-to-API demo. The focus is on correctness, safety, and operability, not only on model accuracy.

---

# 📊 Model Evaluation Report

## 🧠 Model Overview

- **Model type:** `CatBoostRegressor` (CatBoost)
- **Version:** `20260125-2029-9ca500b`

### Target transformation
- **Target:** `log1p`
- **Prediction inverse transform:** `expm1`

---

## 📦 Dataset

| Split | Records |
|------|---------|
| Train | 19,157 |
| Test  | 4,790  |

---

## 📈 Overall Performance (Test Set)

| Metric | Value |
|------|------|
| MdAPE | **11.64 %** |
| WAPE | 17.14 % |
| MAE | 800 166 NOK |
| RMSE | 1 428 602 NOK |
| AE P90 | 1 835 471 NOK |

---

## 🏘️ Performance by Real Estate Type

| Segment | Status | MdAPE (%) | WAPE (%) | AE P90 (NOK) |
|-------|--------|-----------|----------|--------------|
| **Leilighet** | ✅ Good | **8.97 %** | 13.83 % | 1 543 767 |
| **Rekkehus** | ✅ Good | 10.35 % | 14.48 % | 1 489 330 |
| **Tomannsbolig** | ⚠️ OK | 13.23 % | 17.06 % | 1 662 487 |
| **Enebolig** | ⚠️ OK | 17.98 % | 23.28 % | 2 561 397 |
| **Hytte** | 🚨 Risk | **24.95 %** | 32.14 % | 1 953 981 |

---

## 🥇 Best & Worst Segments

- **Best segment:** `leilighet` — MdAPE **8.97%**
- **Worst segment:** `hytte` — MdAPE **24.95%**

---

## 🎯 Quality Thresholds

| Threshold | Value |
|---------|-------|
| **MdAPE (Good)** | ≤ 12% |
| **MdAPE (OK)** | ≤ 18% |
| **AE P90 Tail Risk** | ≥ 2,000,000 NOK |

---

## 📝 Notes

- Worst segment is **`hytte`** (MdAPE 24.95%)
- Likely causes:
  - High price heterogeneity
  - Sparse or noisy data

---

## ⏱️ Metadata

- **Generated at:** `2026-01-25T20:31:53Z`
