import numpy as np
from app.training.metrics import compute_metrics


def test_compute_metrics_smoke():
    y_true = np.array([100.0, 200.0, 400.0])
    y_pred = np.array([110.0, 180.0, 500.0])

    m = compute_metrics(y_true, y_pred)
    assert "mae" in m
    assert "rmse" in m
    assert "mape" in m
    assert "mdape" in m
    assert "ae_p90" in m

    assert m["mae"] > 0
    assert m["rmse"] > 0
    assert m["ae_p90"] > 0
