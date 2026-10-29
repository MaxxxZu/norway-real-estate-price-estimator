import pytest

from app.training.modeling import train_and_evaluate
from tests.test_temporal_evaluation import config, synthetic_rows

pytestmark = pytest.mark.training


def test_train_and_evaluate_smoke():
    result = train_and_evaluate(synthetic_rows(), reference_year=2025, split_config=config())
    assert "mae" in result.metrics["overall"]
    assert result.feature_schema["label"] == "price"
    assert result.metrics["n_validation"] == 2
