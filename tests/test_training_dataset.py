import math

import pytest

from app.training.dataset import build_trainable_dataset
from tests.test_rolling_chain import row


def test_build_trainable_dataset_drops_leilighet_without_floor():
    rows = [
        {
            "id": 1,
            "turnover_date": "2025-01-01",
            "price": 1000000,
            "realestate_type": "leilighet",
            "municipality_number": 301,
            "lat": 59.9,
            "lon": 10.7,
            "built_year": 2000,
            "bra": 50.0,
            "total_area": 55.0,
            "floor": None,
        }
    ]
    res = build_trainable_dataset(rows)
    assert res.trainable_rows == []
    assert res.dropped_reasons.get("missing:floor_for_leilighet") == 1


def test_build_trainable_dataset_accepts_house_without_floor():
    rows = [
        {
            "id": 1,
            "turnover_date": "2025-01-01",
            "price": 1000000,
            "realestate_type": "enebolig",
            "municipality_number": 301,
            "lat": 59.9,
            "lon": 10.7,
            "built_year": 2000,
            "bra": 50.0,
            "total_area": 55.0,
        }
    ]
    res = build_trainable_dataset(rows)
    assert len(res.trainable_rows) == 1


def test_build_trainable_dataset_drops_total_area_lt_bra():
    rows = [
        {
            "id": 1,
            "turnover_date": "2025-01-01",
            "price": 1000000,
            "realestate_type": "rekkehus",
            "municipality_number": 301,
            "lat": 59.9,
            "lon": 10.7,
            "built_year": 2000,
            "bra": 60.0,
            "total_area": 55.0,
        }
    ]
    res = build_trainable_dataset(rows)
    assert res.trainable_rows == []
    assert res.dropped_reasons.get("invalid:total_area_lt_bra") == 1


@pytest.mark.parametrize(
    "field",
    [
        "price",
        "lat",
        "lon",
        "bra",
        "total_area",
        "built_year",
        "floor",
        "bedrooms",
        "rooms",
        "municipality_number",
        "id",
    ],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True, [], {}])
def test_invalid_numeric_values_are_dropped(field, value):
    result = build_trainable_dataset([row(**{field: value})])
    assert result.trainable_rows == []
    assert sum(result.dropped_reasons.values()) == 1


@pytest.mark.parametrize(
    "updates",
    [
        {"id": 0},
        {"built_year": 2000.5},
        {"rooms": -1},
        {"municipality_number": 0},
        {"turnover_date": "invalid"},
        {"turnover_date": 123},
        {"lat": 91},
        {"lon": -181},
    ],
)
def test_domain_constraints(updates):
    assert not build_trainable_dataset([row(**updates)]).trainable_rows


def test_normalized_values_preserved_optional_fields_and_input_unchanged():
    original = row(id="12", lat="59.9", built_year="2001", price="1234567.89")
    result = build_trainable_dataset([original])
    normalized = result.trainable_rows[0]
    assert normalized["id"] == 12 and type(normalized["id"]) is int
    assert normalized["municipality_number"] == 301
    assert normalized["price"] == 1234567.89
    assert normalized["lat"] == 59.9 and normalized["built_year"] == 2001
    assert normalized["turnover_date"] == "2025-01-10"
    assert all(normalized[key] is None for key in ["floor", "bedrooms", "rooms"])
    assert all(
        math.isfinite(value) for value in normalized.values() if isinstance(value, (int, float))
    )
    assert original["id"] == "12" and "rooms" not in original


def test_missing_identity_and_date_have_explanatory_reasons():
    missing_id, missing_date = row(), row()
    del missing_id["id"]
    del missing_date["turnover_date"]
    assert build_trainable_dataset([missing_id, missing_date]).dropped_reasons == {
        "missing:id": 1,
        "missing:turnover_date": 1,
    }
