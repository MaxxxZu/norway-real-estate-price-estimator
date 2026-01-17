from app.training.dataset import build_trainable_dataset


def test_build_trainable_dataset_drops_leilighet_without_floor():
    rows = [
        {
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
