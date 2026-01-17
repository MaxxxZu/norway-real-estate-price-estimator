from app.ml.stub import StubPredictor


def test_stub_predictor_from_artifact():
    p = StubPredictor.from_artifact(
        model_version="stub-v1",
        artifact={"params": {"usable_area_coef": 1, "total_area_coef": 2}},
    )
    assert p.model_version == "stub-v1"
    assert p.usable_area_coef == 1.0
    assert p.total_area_coef == 2.0
