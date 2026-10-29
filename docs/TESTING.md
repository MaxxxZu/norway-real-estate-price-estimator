# Test isolation and Celery execution

Run from the repository root after installing locked dependencies with
`uv sync --locked`. Dependency installation requires package-index access on a
cold cache. Running a command here does not authorize services or model fitting.

## Offline default

```sh
uv run --locked --no-sync pytest
```

The default suite skips `integration` and `training` markers, even when a test is
selected directly or with `-m`. `--strict-markers` rejects misspelled markers.
Before application imports, `tests/conftest.py` clears inherited `REE_*`, `AWS_*`,
`CELERY_*` and legacy API variables, supplies synthetic credentials, disables AWS
metadata/config files, and sets `REE_ENV_FILE` to the null device. Neither `.env`
nor `.env.test` is used. Application dotenv loading still defaults to `.env`;
`REE_ENV_FILE` is an optional explicit path selected before importing Settings.

Offline tests block DNS resolution, IPv4/IPv6 socket connections and CatBoost's fit entry point
during collection and execution. Use fakes/spies or HTTPX mock transports.
These Python guards are regression safeguards, not an OS sandbox: subprocesses
must receive synthetic configuration and must not call services or fit models.
The registration subprocess loads the worker's default modules without starting
a worker, broker, storage client or training task. Worker bootstrap rejection is
also checked in a separate process without starting its consumer.

Real model-fitting tests require separate authorization and explicit selection:

```sh
uv run --locked --no-sync pytest --run-training -m training
```

`--run-training` removes the fit guard; it does not enable network access.
`--run-integration` enables network connections; it does not enable fitting.
No flag by itself constitutes permission to run the associated category.

## Runtime configuration change

Celery includes `app.tasks.rolling` explicitly. Workers consuming `training`
validate their result backend before consuming tasks. The rolling CLI validates
before enqueueing; the bound trigger validates its own app again before creating
a chord. All three paths use the same validator reading `app.conf.result_backend`
(`worker.app` for the worker, the task's app for CLI/trigger). No backend client
is instantiated by this check. Settings supplies the initial app configuration;
validation does not separately re-read the Settings backend.

Precedence follows Celery: a nonempty process `CELERY_RESULT_BACKEND` overrides
app configuration, initially populated from `REE_CELERY_RESULT_BACKEND`. An empty
`CELERY_RESULT_BACKEND` falls back to app configuration. Thus REE Redis + CELERY
RPC is rejected, while REE RPC (or empty) + CELERY Redis is accepted. Configure
the override consistently in both producer and worker environments.
`task_ignore_result=False` is explicit regardless of the Settings backend, so a
supported override retains header results. Validation also rejects an effective
`task_ignore_result=True` configuration. Fresh-process tests cover conflicting
overrides for worker bootstrap, CLI and trigger, with connection guards and
fake dispatch; they do not start consumers or contact services.

Only the project's Redis backends (`redis://`, `rediss://`, `redis+socket://`)
are accepted. Errors omit URLs and credentials. This is a configuration check,
not a Redis connectivity/readiness check. Inference settings/application imports
do not perform training-backend validation.

The result backend default and public example now use
`redis://localhost:6379/0` instead of `rpc://`. Existing personal dotenv files
are not migrated: set `REE_CELERY_RESULT_BACKEND` explicitly to Redis for rolling
workers/producers. Install the new lockfile. Compose provides Redis on port 6379
and RabbitMQ on port 5672 for host-run workers. For workers in containers, use
the service hostname `redis` instead of `localhost`. Validate deployment-specific Helm/backend overrides separately before running
training workers.

## Actual RabbitMQ/Redis chord integration

Requires explicit authorization to start local services and run integration tests.
Use disposable local services and an otherwise unused Redis database (default 15).
The test rejects remote hosts and does not read personal service URLs. The
following commands start only RabbitMQ and Redis, not MinIO or training schedules:

```sh
docker compose --env-file /dev/null -p ree-chord-test up -d --wait rabbitmq redis
uv run --locked --no-sync pytest --run-integration -m integration tests/integration/test_chord.py
# Stop only this disposable project's two services after review:
docker compose --env-file /dev/null -p ree-chord-test stop rabbitmq redis
```

Optional `TEST_CELERY_BROKER_URL` / `TEST_CELERY_RESULT_BACKEND` overrides must
point to disposable loopback RabbitMQ/Redis, not personal or production services.
Host ports must be free; do not stop unrelated services to free them automatically.

The scenario starts a separate Celery worker process with a unique queue and
`task_always_eager=False`. Three synthetic header tasks return values through
Redis; the callback receives all three and returns a checked aggregate. Its
45-second result timeout fails on missing worker registration/chord completion.
The worker is terminated and the unique queue removed in `finally`; completed
results are forgotten. A failed run may leave expiring Redis result keys. Never
flush a shared Redis database. Worker logs are retained in pytest's temporary
directory for local diagnosis.

This exercises actual broker/backend chord transport with the production Celery
configuration and rolling module include. It does not execute the real ingestion,
training, gating or publication tasks, prove prefork failure handling, or establish
storage compatibility. Eager tests are not integration evidence. A passing smoke run establishes only
the tested transport configuration, not general worker reliability.


## Recorded synthetic chord smoke — 2026-10-28

The integration test passed: `1 passed in 12.98s`, with Celery 5.6.2,
Python 3.11.11, RabbitMQ 3.13.7 and Redis 7.4.9 on Docker Desktop linux/arm64.
An isolated temporary Compose file copied only RabbitMQ/Redis services and
healthchecks from this repository, publishing ports exclusively on 127.0.0.1.
The test used a unique project and queue, a separate worker process with pool
`solo` / concurrency 1, AMQP on port 5672 and Redis database 15 on port 6379.
No eager execution, model fitting, data API, MinIO or training pipeline was used.

Worker logs confirmed all three header tasks succeeded with values 3, 1 and 2,
and the callback succeeded with `{"values": [1, 2, 3], "total": 6}`. The test's
`result.get()` obtained the checked callback result through Redis. The worker
was absent after teardown and RabbitMQ reported no remaining queues.

Cleanup used `docker compose ... down --volumes` against only the unique
synthetic project: its two containers, two anonymous volumes and network were
removed, and ports 5672/15672/6379 were free again. Images remain in Docker's
cache. The recorded outcome describes the local configuration used for this smoke.
This is one successful transport smoke; prefork operation, retries/redelivery,
broker/backend outages and real rolling ingestion/training/publication remain
outside this evidence. The smoke does not establish end-to-end training or
production deployment readiness.
