"""Explicit calendar partitions and auditable exposure membership."""

from collections import Counter
from dataclasses import dataclass
from datetime import date
from typing import Any

from app.training.dataset import build_trainable_dataset
from app.training.snapshots import semantic_fingerprint


@dataclass(frozen=True)
class TemporalSplitConfig:
    train_start: date
    validation_start: date
    test_start: date
    test_end: date  # exclusive

    def __post_init__(self):
        values = (self.train_start, self.validation_start, self.test_start, self.test_end)
        if any(type(value) is not date for value in values) or not all(
            a < b for a, b in zip(values, values[1:])
        ):
            raise ValueError("Require train_start < validation_start < test_start < test_end")

    def as_dict(self):
        return {key: value.isoformat() for key, value in vars(self).items()}

    @classmethod
    def from_settings(cls):
        from app.config import settings

        return cls(
            *(
                getattr(settings, f"split_{key}")
                for key in ("train_start", "validation_start", "test_start", "test_end")
            )
        )


@dataclass(frozen=True)
class TemporalSplit:
    train: list[dict[str, Any]]
    validation: list[dict[str, Any]]
    test: list[dict[str, Any]]
    metadata: dict[str, Any]


def membership(rows):
    return sorted(
        [
            {
                "id": row["id"],
                "date": row["turnover_date"],
                "row_hash": semantic_fingerprint([row]),
            }
            for row in rows
        ],
        key=lambda entry: (entry["date"], entry["id"], entry["row_hash"]),
    )


def split_rows(rows, config: TemporalSplitConfig, dataset_identity=None) -> TemporalSplit:
    normalized = build_trainable_dataset(rows)
    if normalized.dropped_reasons or not normalized.trainable_rows:
        raise ValueError(f"Split requires valid rows: {normalized.dropped_reasons}")
    rows = sorted(
        normalized.trainable_rows,
        key=lambda row: (row["turnover_date"], row["id"], semantic_fingerprint([row])),
    )
    boundaries = list(config.as_dict().values())
    if any(not boundaries[0] <= row["turnover_date"] < boundaries[-1] for row in rows):
        raise ValueError("Rows outside configured calendar window")
    parts = {
        name: [row for row in rows if start <= row["turnover_date"] < end]
        for name, start, end in zip(("train", "validation", "test"), boundaries, boundaries[1:])
    }
    if any(not part for part in parts.values()):
        raise ValueError("All temporal partitions must be nonempty")
    ids = {name: {row["id"] for row in part} for name, part in parts.items()}
    overlaps = {
        f"{a}:{b}": sorted(ids[a] & ids[b])
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }
    if any(overlaps.values()):
        raise ValueError("Property IDs overlap across temporal partitions")
    if len({row["price"] for row in parts["train"]}) < 2:
        raise ValueError("CatBoost RMSE train requires at least two distinct labels")
    metadata = {
        "version": 1,
        "training_mode": "fresh_fit",
        "boundaries": config.as_dict(),
        "property_overlap_policy": "reject_cross_partition",
        "overlaps": overlaps,
        "dataset": {"fingerprint": semantic_fingerprint(rows), "source": dataset_identity},
        "partitions": {
            name: {
                "count": len(part),
                "segments": dict(sorted(Counter(row["realestate_type"] for row in part).items())),
                "period": {"start": part[0]["turnover_date"], "end": part[-1]["turnover_date"]},
                "fingerprint": semantic_fingerprint(part),
                "membership": membership(part),
                "membership_fingerprint": semantic_fingerprint(membership(part)),
            }
            for name, part in parts.items()
        },
    }
    return TemporalSplit(**parts, metadata=metadata)
