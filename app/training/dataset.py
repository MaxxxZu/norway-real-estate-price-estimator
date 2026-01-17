from collections import Counter
from dataclasses import dataclass
from typing import Any

from app.schemas import RealEstateType


REQUIRED_FIELDS = [
    "price",
    "realestate_type",
    "municipality_number",
    "lat",
    "lon",
    "built_year",
    "bra",
    "total_area",
]


@dataclass(frozen=True)
class DatasetBuildResult:
    trainable_rows: list[dict[str, Any]]
    dropped_reasons: dict[str, int]


def build_trainable_dataset(rows: list[dict[str, Any]]) -> DatasetBuildResult:
    reasons = Counter()
    trainable: list[dict[str, Any]] = []

    for r in rows:
        missing = [f for f in REQUIRED_FIELDS if r.get(f) is None]
        if missing:
            reasons[f"missing:{','.join(missing)}"] += 1
            continue

        if not isinstance(r["price"], (int, float)) or r["price"] <= 0:
            reasons["invalid:price"] += 1
            continue

        if not isinstance(r["bra"], (int, float)) or r["bra"] <= 0:
            reasons["invalid:bra"] += 1
            continue

        if not isinstance(r["total_area"], (int, float)) or r["total_area"] <= 0:
            reasons["invalid:total_area"] += 1
            continue

        if float(r["total_area"]) < float(r["bra"]):
            reasons["invalid:total_area_lt_bra"] += 1
            continue

        rt = r.get("realestate_type")
        try:
            rt_enum = RealEstateType(rt)
        except Exception:
            reasons["invalid:realestate_type"] += 1
            continue

        if rt_enum == RealEstateType.leilighet and r.get("floor") is None:
            reasons["missing:floor_for_leilighet"] += 1
            continue

        try:
            lat = float(r["lat"])
            lon = float(r["lon"])
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                raise ValueError
        except Exception:
            reasons["invalid:latlon"] += 1
            continue

        try:
            by = int(r["built_year"])
            if by < 1800 or by > 2100:
                raise ValueError
        except Exception:
            reasons["invalid:built_year"] += 1
            continue

        trainable.append(r)

    return DatasetBuildResult(trainable_rows=trainable, dropped_reasons=dict(reasons))
