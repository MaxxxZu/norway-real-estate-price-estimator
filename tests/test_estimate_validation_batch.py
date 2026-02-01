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
    errors = body["detail"]
    assert isinstance(errors, list)
    error_locs = [e["loc"] for e in errors]
    assert any("bad_missing" in str(loc) for loc in error_locs)
    assert any("bad_logic" in str(loc) for loc in error_locs)
    assert not any("ok" in str(loc) for loc in error_locs)
