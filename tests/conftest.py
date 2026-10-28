import os
import socket

import pytest
from fastapi.testclient import TestClient

# Run before test-module imports: never load personal dotenv files or credentials.
for key in list(os.environ):
    if key.startswith(("REE_", "AWS_", "CELERY_")) or key in {"API_KEY", "API_BASE_URL"}:
        del os.environ[key]
os.environ.update(
    REE_ENV_FILE=os.devnull,
    REE_ENV="test",
    REE_API_BASE_URL="http://data-api.invalid",
    REE_API_KEY="synthetic-test-key",
    REE_S3_ACCESS_KEY="synthetic-access-key",
    REE_S3_SECRET_KEY="synthetic-secret-key",
    REE_S3_ENDPOINT="http://storage.invalid",
    AWS_EC2_METADATA_DISABLED="true",
    AWS_CONFIG_FILE=os.devnull,
    AWS_SHARED_CREDENTIALS_FILE=os.devnull,
    BOTO_CONFIG=os.devnull,
)


def pytest_addoption(parser):
    parser.addoption("--run-integration", action="store_true", help="Allow local service tests")
    parser.addoption("--run-training", action="store_true", help="Allow real model fitting")


def pytest_collection_modifyitems(config, items):
    for item in items:
        for marker in ("integration", "training"):
            if item.get_closest_marker(marker) and not config.getoption(f"--run-{marker}"):
                item.add_marker(pytest.mark.skip(reason=f"requires --run-{marker}"))


def pytest_configure(config):
    # Guard collection as well as execution, including accidentally unmarked tests.
    guard = pytest.MonkeyPatch()
    config._offline_guard = guard
    if not config.getoption("--run-integration"):
        original_connect = socket.socket.connect

        def offline_connect(sock, address):
            if sock.family in {socket.AF_INET, socket.AF_INET6}:
                raise RuntimeError("Network disabled in offline tests; use a fake transport")
            return original_connect(sock, address)

        def no_dns(*args, **kwargs):
            raise RuntimeError("Network disabled in offline tests; use a fake transport")

        guard.setattr(socket, "getaddrinfo", no_dns)
        guard.setattr(socket.socket, "connect", offline_connect)
        guard.setattr(socket.socket, "connect_ex", offline_connect)
    if not config.getoption("--run-training"):
        from catboost import CatBoost

        def no_fit(*args, **kwargs):
            raise RuntimeError("Model fitting disabled; requires --run-training")

        guard.setattr(CatBoost, "_fit", no_fit)


def pytest_unconfigure(config):
    guard = getattr(config, "_offline_guard", None)
    if guard:
        guard.undo()


@pytest.fixture(scope="session")
def client():
    from app.main import app
    from app.ml.stub import StubPredictor
    from app.routes import get_predictor

    stub = StubPredictor.from_artifact(
        model_version="stub-v1",
        artifact={"params": {"usable_area_coef": 50_000, "total_area_coef": 5_000}},
    )
    app.dependency_overrides[get_predictor] = lambda: stub

    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
