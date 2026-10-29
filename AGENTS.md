# Repository guidelines

## Project structure

A FastAPI service for CatBoost property-price estimation, with Celery training
workflows and S3-compatible data/model storage. Preserve the existing stack and
module boundaries unless the assigned change requires otherwise.

- `app.main`, `app.routes`, `app.schemas`: HTTP application and request contracts.
- `app.services`, `app.ml`: estimation service, predictors and model registry.
- `app.training`: ingestion, datasets, evaluation, gating and artifact publication.
- `app.tasks`, `app.celery_app`: background workflow and worker configuration.
- `app.storage`, `app.observability`: storage adapters, logging and metrics.
- `scripts`, `deploy`, `.github/workflows`: entry points, deployment and CI.
- `tests`: offline, integration and model-fitting checks.

Public behavior and compatibility contracts are documented in README.md and docs/.
Update relevant documentation when changing a contract.

## Engineering conventions

- Inspect the current implementation and Git status before editing; preserve
  unrelated changes. Keep the assigned scope small and reviewable.
- Keep HTTP, model inference, training and storage responsibilities separate.
  Share core contracts between synchronous and Celery entry points.
- Preserve deterministic data/feature transformations and explicit schema versions.
  Do not silently reinterpret old artifacts, guess metadata or delete historical data.
- Keep model evaluation separate from fitting; validation controls early stopping.
  Record lineage and use comparable holdouts for model comparisons.
- Do not introduce new services or broad refactors without a concrete need.
- Use English for code, comments and public documentation.

## Validation and safety

- Use synthetic fixtures and isolated fake transports/storage for offline tests.
  Follow docs/TESTING.md for test modes; some tests fit models or contact services.
- Run relevant lint/format and regression checks within the task's authorized scope.
  Report the checks actually run, unrun checks and compatibility implications.
- Test orchestration and failure boundaries, not just implementation details.
  Spies do not establish real model quality, and one provider's tests do not prove
  another provider's compatibility.
- Never expose credentials or publish data/model artifacts without confirmed rights.
  Treat joblib models as trusted executable inputs; hashes do not authenticate publishers.
- Do not modify real environment files, external resources or Git history without
  explicit authorization. Respect any additional execution limits in the task.

## Public documentation

- Separate implemented behavior, verified evidence and known limitations.
  Do not invent benchmark results, licenses or production-readiness guarantees.
- Use synthetic examples or examples with established publication rights;
  do not assume existing identifiers or precise coordinates are synthetic.
- Keep personal paths, credentials, raw private logs and internal working notes
  out of tracked content. Public setup instructions must be self-contained.
- Explain breaking changes and keep historical metrics distinct from current claims.
