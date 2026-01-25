from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    abs_err = np.abs(y_true - y_pred)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    denom_mape = np.where(np.abs(y_true) > 1e-9, np.abs(y_true), np.nan)
    mape = float(np.nanmean(abs_err / denom_mape))
    mdape = float(np.nanmedian(abs_err / denom_mape))

    denom_wape = float(np.sum(np.abs(y_true)))
    wape = float(np.sum(abs_err) / denom_wape) if denom_wape > 0 else float("nan")

    ae_p90 = float(np.quantile(abs_err, 0.90))

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "mdape": mdape,
        "ae_p90": ae_p90,
        "wape": wape,
    }
