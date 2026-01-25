import os

import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient

load_dotenv('.env.test', override=True)


@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    required_vars = ['API_BASE_URL', 'API_KEY']
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise RuntimeError(f"Missing required env vars for tests: {missing}")
    yield


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
