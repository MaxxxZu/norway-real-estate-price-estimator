def test_batch_validation_errors_are_per_property(client):
    payload = {
        "ok": {
            "realestate_type": "enebolig",
            "municipality_number": 4003,
            "lat": 59.21,
            "lon": 9.58,
            "built_year": 2016,
            "total_area": 200.0,
            "bra": 150.0,
        },
        "bad_missing": {"realestate_type": "enebolig"},
        "bad_logic": {
            "realestate_type": "rekkehus",
            "municipality_number": 1,
            "lat": 59.0,
            "lon": 10.0,
            "built_year": 2000,
            "total_area": 50.0,
            "bra": 60.0,
        },
    }

    resp = client.post("/estimate", json=payload)
    assert resp.status_code == 422
    body = resp.json()
    assert "detail" in body
    detail = body["detail"]
    assert isinstance(detail, dict)
    assert detail.get("message") == "validation_failed"
    errors = detail.get("errors")
    assert isinstance(errors, dict)
    assert "bad_missing" in errors
    assert "bad_logic" in errors
    assert "ok" not in errors
