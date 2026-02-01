from datetime import UTC, datetime
from typing import Any

from app.config import settings


def _status_from_mdape(mdape: float) -> str:
    if mdape <= settings.metrics_mdape_good_threshold:
        return "good"
    if mdape <= settings.metrics_mdape_ok_threshold:
        return "ok"
    return "risk"


def build_metrics_summary(raw: dict[str, Any]) -> dict[str, Any]:
    metrics = raw["metrics"]

    overall = metrics["overall"]
    by_type = metrics.get("by_realestate_type", {})

    segments: list[dict[str, Any]] = []
    for key, m in by_type.items():
        segments.append(
            {
                "key": key,
                "status": _status_from_mdape(float(m["mdape"])),
                "mdape_pct": round(float(m["mdape"]) * 100, 2),
                "wape_pct": round(float(m["wape"]) * 100, 2),
                "ae_p90_nok": int(float(m["ae_p90"])),
            }
        )

    segments_sorted = sorted(segments, key=lambda x: x["mdape_pct"])
    best = segments_sorted[0] if segments_sorted else None
    worst = segments_sorted[-1] if segments_sorted else None

    notes: list[str] = []
    if worst and worst["status"] == "risk":
        notes.append(
            f"Worst segment is '{worst['key']}' (MdAPE {worst['mdape_pct']}%). "
            "Likely heterogeneous pricing or sparse data."
        )

    if int(float(overall["ae_p90"])) >= settings.metrics_ae_p90_tail_risk_nok:
        notes.append(f"Tail risk: AE_p90 is ~{int(float(overall['ae_p90'])):,} NOK overall.")

    return {
        "model": {
            "version": raw["model_version"],
            "type": raw["model_type"],
            "family": metrics.get("model_family"),
            "target_transform": metrics.get("target_transform"),
            "prediction_transform": metrics.get("prediction_transform"),
        },
        "data": {
            "n_train": int(metrics["n_train"]),
            "n_test": int(metrics["n_test"]),
        },
        "overall": {
            "mdape_pct": round(float(overall["mdape"]) * 100, 2),
            "wape_pct": round(float(overall["wape"]) * 100, 2),
            "mae_nok": int(float(overall["mae"])),
            "rmse_nok": int(float(overall["rmse"])),
            "ae_p90_nok": int(float(overall["ae_p90"])),
        },
        "segments": {
            "by_realestate_type": segments_sorted,
            "best": {"key": best["key"], "mdape_pct": best["mdape_pct"]} if best else None,
            "worst": {"key": worst["key"], "mdape_pct": worst["mdape_pct"]} if worst else None,
        },
        "thresholds": {
            "mdape_good": settings.metrics_mdape_good_threshold,
            "mdape_ok": settings.metrics_mdape_ok_threshold,
            "ae_p90_tail_risk_nok": settings.metrics_ae_p90_tail_risk_nok,
        },
        "notes": notes,
        "generated_at": datetime.now(UTC).isoformat(),
    }
