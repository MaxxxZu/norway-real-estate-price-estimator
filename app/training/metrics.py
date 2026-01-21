import numpy as np


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    ae = np.abs(y_true - y_pred)
    mae = float(np.mean(ae))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    mask = y_true > 0
    if np.any(mask):
        ape = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
        mape = float(np.mean(ape))
        mdape = float(np.median(ape))
    else:
        mape = float("nan")
        mdape = float("nan")

    ae_p90 = float(np.quantile(ae, 0.90))

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "mdape": mdape,
        "ae_p90": ae_p90,
    }
