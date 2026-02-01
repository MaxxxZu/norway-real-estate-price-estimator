from dataclasses import dataclass
from typing import Any

from app.config import settings


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reasons: list[str]
    details: dict[str, Any]


def _get_metric(metrics: dict[str, Any], path: list[str]) -> float | None:
    cur: Any = metrics
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    try:
        return float(cur)
    except (TypeError, ValueError):
        return None


def _pct_change(new: float, prev: float) -> float:
    if prev == 0:
        return float("inf") if new > 0 else 0.0
    return (new - prev) / prev


def _check_degradation(
    *,
    name: str,
    new_val: float | None,
    prev_val: float | None,
    max_degrade_pct: float,
    reasons: list[str],
    checks: dict[str, Any],
) -> None:
    checks[name] = {
        "new": new_val,
        "prev": prev_val,
        "max_degrade_pct": max_degrade_pct,
    }
    if new_val is None or prev_val is None:
        checks[name]["status"] = "missing_metric"
        return

    degrade = _pct_change(new_val, prev_val)
    checks[name]["degrade_pct"] = degrade

    if degrade > max_degrade_pct:
        reasons.append(f"degraded:{name}:{degrade:.3f} > {max_degrade_pct:.3f}")
        checks[name]["status"] = "failed"
    else:
        checks[name]["status"] = "passed"


def evaluate_publish_gate(
    *,
    rows_trainable: int,
    new_metrics: dict[str, Any],
    prev_metrics: dict[str, Any] | None,
) -> GateResult:
    reasons: list[str] = []
    details: dict[str, Any] = {
        "rows_trainable": rows_trainable,
        "min_rows_required": settings.train_min_rows,
        "has_prev": prev_metrics is not None,
        "rules": {
            "overall_max_degrade_pct": settings.gate_overall_mae_max_degrade_pct,
            "overall_wape_max_degrade_pct": settings.gate_overall_wape_max_degrade_pct,
            "enebolig_max_degrade_pct": settings.gate_enebolig_mae_max_degrade_pct,
            "enebolig_wape_max_degrade_pct": settings.gate_enebolig_wape_max_degrade_pct,
        },
        "checks": {},
    }

    if rows_trainable < settings.train_min_rows:
        reasons.append("insufficient_rows_trainable")
        details["checks"]["min_rows"] = {"status": "failed"}
        return GateResult(passed=False, reasons=reasons, details=details)

    details["checks"]["min_rows"] = {"status": "passed"}
    if prev_metrics is None:
        details["checks"]["comparison"] = {"status": "passed_no_prev"}
        return GateResult(passed=True, reasons=[], details=details)

    checks: dict[str, Any] = {}
    _check_degradation(
        name="overall.mdape",
        new_val=_get_metric(new_metrics, ["overall", "mdape"]),
        prev_val=_get_metric(prev_metrics, ["overall", "mdape"]),
        max_degrade_pct=settings.gate_overall_mae_max_degrade_pct,
        reasons=reasons,
        checks=checks,
    )
    _check_degradation(
        name="overall.ae_p90",
        new_val=_get_metric(new_metrics, ["overall", "ae_p90"]),
        prev_val=_get_metric(prev_metrics, ["overall", "ae_p90"]),
        max_degrade_pct=settings.gate_overall_mae_max_degrade_pct,
        reasons=reasons,
        checks=checks,
    )
    _check_degradation(
        name="overall.wape",
        new_val=_get_metric(new_metrics, ["overall", "wape"]),
        prev_val=_get_metric(prev_metrics, ["overall", "wape"]),
        max_degrade_pct=settings.gate_overall_wape_max_degrade_pct,
        reasons=reasons,
        checks=checks,
    )
    _check_degradation(
        name="enebolig.mdape",
        new_val=_get_metric(new_metrics, ["by_realestate_type", "enebolig", "mdape"]),
        prev_val=_get_metric(prev_metrics, ["by_realestate_type", "enebolig", "mdape"]),
        max_degrade_pct=settings.gate_enebolig_mae_max_degrade_pct,
        reasons=reasons,
        checks=checks,
    )
    _check_degradation(
        name="enebolig.ae_p90",
        new_val=_get_metric(new_metrics, ["by_realestate_type", "enebolig", "ae_p90"]),
        prev_val=_get_metric(prev_metrics, ["by_realestate_type", "enebolig", "ae_p90"]),
        max_degrade_pct=settings.gate_enebolig_mae_max_degrade_pct,
        reasons=reasons,
        checks=checks,
    )
    _check_degradation(
        name="enebolig.wape",
        new_val=_get_metric(new_metrics, ["by_realestate_type", "enebolig", "wape"]),
        prev_val=_get_metric(prev_metrics, ["by_realestate_type", "enebolig", "wape"]),
        max_degrade_pct=settings.gate_enebolig_wape_max_degrade_pct,
        reasons=reasons,
        checks=checks,
    )

    passed = len(reasons) == 0
    details["checks"]["comparison"] = {
        "status": "passed" if passed else "failed",
        "checks": checks,
    }

    return GateResult(passed=passed, reasons=reasons, details=details)
