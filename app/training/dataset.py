"""Versioned, normalized rows; rejected values never reach the feature builder."""

from collections import Counter
from dataclasses import dataclass
from datetime import date
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from app.schemas import EstimationFeatures

NORMALIZATION_VERSION = 1


class TrainingIdentity(BaseModel):
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)

    id: int = Field(gt=0)
    turnover_date: date

    @model_validator(mode="before")
    @classmethod
    def reject_booleans(cls, values):
        if not isinstance(values, dict):
            raise ValueError("row must be an object")
        for key in cls.model_fields:
            if isinstance(values.get(key), bool):
                raise ValueError(f"boolean:{key}")
        return values

    @field_validator("turnover_date", mode="before")
    @classmethod
    def parse_date(cls, value):
        if isinstance(value, str):
            return date.fromisoformat(value)
        if type(value) is date:
            return value
        raise ValueError("expected ISO calendar date")


class TrainingRow(TrainingIdentity, EstimationFeatures):
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)
    price: float = Field(gt=0)


@dataclass(frozen=True)
class DatasetBuildResult:
    trainable_rows: list[dict[str, Any]]
    dropped_reasons: dict[str, int]


def build_trainable_dataset(rows: list[dict[str, Any]]) -> DatasetBuildResult:
    reasons = Counter()
    trainable = []
    for row in rows:
        try:
            normalized = TrainingRow.model_validate(row)
        except ValidationError as exc:
            error = exc.errors()[0]
            field = ".".join(map(str, error["loc"])) or "row"
            message = error["msg"]
            if "floor is required" in message:
                reason = "missing:floor_for_leilighet"
            elif "total_area must be >= bra" in message:
                reason = "invalid:total_area_lt_bra"
            elif "boolean:" in message:
                reason = "invalid:" + message.split("boolean:", 1)[1]
            else:
                reason = f"{'missing' if error['type'] == 'missing' else 'invalid'}:{field}"
            reasons[reason] += 1
            continue
        trainable.append(normalized.model_dump(mode="json"))
    return DatasetBuildResult(trainable, dict(sorted(reasons.items())))
