from app.training.modeling import train_and_evaluate


def test_train_and_evaluate_smoke():
    data = [
        {
            "price": 3_000_000,
            "realestate_type": "enebolig",
            "municipality_number": 301,
            "lat": 59.9,
            "lon": 10.7,
            "built_year": 2000,
            "bra": 120.0,
            "total_area": 200.0,
            "floor": None,
            "bedrooms": 3,
            "rooms": 5,
        },
        {
            "price": 4_000_000,
            "realestate_type": "leilighet",
            "municipality_number": 301,
            "lat": 59.91,
            "lon": 10.75,
            "built_year": 2010,
            "bra": 70.0,
            "total_area": 75.0,
            "floor": 3,
            "bedrooms": 2,
            "rooms": 3,
        },
        {
            "price": 2_500_000,
            "realestate_type": "rekkehus",
            "municipality_number": 4003,
            "lat": 59.2,
            "lon": 9.58,
            "built_year": 1995,
            "bra": 90.0,
            "total_area": 110.0,
            "floor": None,
            "bedrooms": 2,
            "rooms": 4,
        },
        {
            "price": 5_500_000,
            "realestate_type": "tomannsbolig",
            "municipality_number": 5001,
            "lat": 63.43,
            "lon": 10.39,
            "built_year": 2018,
            "bra": 140.0,
            "total_area": 220.0,
            "floor": None,
            "bedrooms": 3,
            "rooms": 6,
        },
        {
            "price": 1_800_000,
            "realestate_type": "hytte",
            "municipality_number": 4223,
            "lat": 58.3,
            "lon": 7.9,
            "built_year": 1980,
            "bra": 55.0,
            "total_area": 80.0,
            "floor": None,
            "bedrooms": 1,
            "rooms": 2,
        },
    ]

    rows = data * 20

    res = train_and_evaluate(rows)
    assert "overall" in res.metrics
    assert "mae" in res.metrics["overall"]
    assert res.feature_schema["label"] == "price"
