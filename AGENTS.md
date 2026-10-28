## Project and purpose

Norway Real Estate Price Estimator is a production-oriented portfolio project
for Python Backend / Backend & AI Engineer roles. Improve engineering correctness
and provide verifiable evidence; do not claim production readiness from the
presence of infrastructure files alone.

Communicate with the owner in Russian. Keep code, comments, commit messages and
repository documentation in English unless the owner requests otherwise.

Keep FastAPI, CatBoost, Celery, RabbitMQ, Redis result backend and S3-compatible
storage. Preserve the architecture where reasonable. Do not rewrite the project,
introduce an enterprise platform, or perform unrelated cosmetic refactoring.

## Working agreement

- Read these instructions, inspect Git status and the current implementation
  before making changes. Preserve user changes and unrelated work.
- Implement one explicitly assigned task or smaller slice at a time.
  Do not advance to other tasks automatically.
- Use a dedicated branch for implementation. Never commit or push; the owner
  performs these actions. Do not open a PR, merge, publish artifacts or deploy
  unless the user authorizes that action.
- Ask only for genuinely missing requirements, consequential policy decisions
  or required permissions. Existing user authorization persists in the session;
  do not ask repeatedly for the same permitted action.
- Do not silently choose business quality thresholds, data rights, a license,
  public API exposure, legacy migration or production activation policies.
- Keep changes small and reviewable. Document contract changes and migration
  implications. Do not delete old models, datasets or snapshots automatically.
- Finish with changed behavior, affected files, checks actually performed,
  unperformed checks, compatibility risks and remaining work. Stop at the
  assigned scope boundary.

## Execution and security boundaries

Permission to edit code does not override an explicit restriction on tests,
network access or infrastructure. Until separately authorized, do not:

- run tests, including tests that fit a CatBoost model;
- install/update dependencies or regenerate lockfiles using network resolution;
- launch Docker, Kubernetes, deployment or scheduled jobs;
- call external APIs, train/evaluate on real data, or write to object storage;
- activate a model, execute rollback, rotate credentials or rewrite Git history.

Once the user authorizes a category, execute relevant checks within that scope
without asking again. Respect environment approval requirements. If a check is
not authorized or unavailable, report it as unverified rather than passed.

Never expose credentials or secret values. Do not edit real `.env`, `.env.kind`
or `.env.gke` files without an explicit request. Use synthetic fixtures and
redacted output. Do not publish real datasets/models without confirmed rights.
Checksums establish integrity relative to trusted metadata, not publisher trust.

## Existing architecture

- `app.main`, `app.routes`, `app.schemas`: FastAPI application and API contracts.
- `app.services.estimate_service`: estimation orchestration.
- `app.ml`: predictor interface, implementations, registry and model metrics.
- `app.clients.api_client`: external data API client.
- `app.training`: ingestion, normalization, snapshots, features/modeling,
  evaluation, gating and publication.
- `app.tasks.rolling`, `app.celery_app`: asynchronous rolling training workflow.
- `app.storage.s3`: S3-compatible storage adapter.
- `app.observability`: request IDs, logging and Prometheus instrumentation.
- `scripts`: synchronous and asynchronous CLI entry points.
- `deploy`, `.github/workflows`: infrastructure and CI/release configuration.
- `tests`: existing tests; some execute real model fitting.

Synchronous and Celery paths should share contracts and core services instead of
developing different evaluation, gating and publication semantics.

## Validation and reporting

For each authorized implementation slice: review the diff, run relevant Ruff
lint/format checks and targeted/regression tests, and document changed contracts.
Run the full authorized offline suite after shared-contract changes and before
final acceptance. Avoid unrelated cleanup or mass dependency updates.

Use spies/fakes for unit-level split, gate, registry and publication failures.
Actual Redis/RabbitMQ chords, storage conditional writes, container startup and
kind deployment require integration evidence. Synthetic CatBoost round trips
still execute training and need the corresponding permission. GCS compatibility
requires its own evidence; MinIO tests do not establish it.

Do not invent model-quality numbers, production thresholds, coverage results or
deployment success. Synthetic demo results are not real-world model evaluation.
Portfolio readiness means reproducible operation, credible evaluation, tested
failure handling and accurate documentation; a live cloud deployment is optional.
