import pytest
from fastapi.testclient import TestClient


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
