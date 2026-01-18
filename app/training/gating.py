from dataclasses import dataclass
from typing import Any

from app.config import settings


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    reasons: list[str]
    details: dict[str, Any]


def _pct_degrade(prev: float, new: float) -> float | None:
    if prev <= 0:
        return None
    return (new - prev) / prev


def evaluate_publish_gate(
    *,
    rows_trainable: int,
    new_metrics: dict[str, Any],
    prev_metrics: dict[str, Any] | None,
) -> GateDecision:
    reasons: list[str] = []
    details: dict[str, Any] = {"thresholds": {}}

    min_rows = int(settings.train_min_rows)
    details["thresholds"]["train_min_rows"] = min_rows
    details["rows_trainable"] = int(rows_trainable)

    if rows_trainable < min_rows:
        reasons.append(f"min_rows_failed: rows_trainable={rows_trainable} < {min_rows}")

    if not prev_metrics:
        return GateDecision(passed=(len(reasons) == 0), reasons=reasons, details=details)

    try:
        prev_overall_mae = float(prev_metrics["overall"]["mae"])
        new_overall_mae = float(new_metrics["overall"]["mae"])
        details["prev_overall_mae"] = prev_overall_mae
        details["new_overall_mae"] = new_overall_mae

        degrade = _pct_degrade(prev_overall_mae, new_overall_mae)
        details["overall_mae_degrade_pct"] = degrade

        max_degrade = float(settings.gate_overall_mae_max_degrade_pct)
        details["thresholds"]["overall_mae_max_degrade_pct"] = max_degrade

        if degrade is not None and degrade > max_degrade:
            reasons.append(
                f"overall_mae_degraded: {degrade:.2%} > {max_degrade:.2%}"
            )
    except Exception:
        reasons.append("overall_mae_compare_failed")

    try:
        prev_by_type = prev_metrics.get("by_realestate_type_mae", {}) or {}
        new_by_type = new_metrics.get("by_realestate_type_mae", {}) or {}

        prev_enebolig = prev_by_type.get("enebolig")
        new_enebolig = new_by_type.get("enebolig")

        details["prev_enebolig_mae"] = prev_enebolig
        details["new_enebolig_mae"] = new_enebolig

        if prev_enebolig is not None and new_enebolig is not None:
            prev_enebolig = float(prev_enebolig)
            new_enebolig = float(new_enebolig)

            degrade = _pct_degrade(prev_enebolig, new_enebolig)
            details["enebolig_mae_degrade_pct"] = degrade

            max_degrade = float(settings.gate_enebolig_mae_max_degrade_pct)
            details["thresholds"]["enebolig_mae_max_degrade_pct"] = max_degrade

            if degrade is not None and degrade > max_degrade:
                reasons.append(
                    f"enebolig_mae_degraded: {degrade:.2%} > {max_degrade:.2%}"
                )
        else:
            details["enebolig_compare_skipped"] = True
    except Exception:
        reasons.append("enebolig_mae_compare_failed")

    return GateDecision(passed=(len(reasons) == 0), reasons=reasons, details=details)
