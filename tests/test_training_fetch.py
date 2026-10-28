import pytest

from app.training.fetch import build_rows, normalize_turnovers


@pytest.mark.parametrize("price", [float("nan"), float("inf"), -float("inf"), True, "123"])
def test_invalid_turnover_price_rejected(price):
    assert (
        normalize_turnovers(
            [
                {
                    "price": price,
                    "cadastral_unit_ids": [1],
                    "turnover_date": "2025-01-01T12:00:00Z",
                }
            ]
        )
        == []
    )


def test_turnover_price_precision_and_authoritative_identity_preserved():
    turnovers = normalize_turnovers(
        [
            {
                "price": 1234567.89,
                "cadastral_unit_ids": [1],
                "turnover_date": "2025-01-01T12:00:00Z",
            }
        ]
    )
    rows = build_rows(
        turnovers,
        {1: {"property_ids": [2], "full_unit": "synthetic"}},
        {"2": {"id": 999, "price": 1, "turnover_date": "1900-01-01", "bra": 50}},
    )
    assert rows[0]["id"] == 2
    assert rows[0]["price"] == 1234567.89
    assert rows[0]["turnover_date"] == "2025-01-01"
    assert rows[0]["bra"] == 50
