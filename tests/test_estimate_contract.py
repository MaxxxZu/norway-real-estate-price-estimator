def test_estimate_ok_house(client):
    payload = {
        "486002054": {
            "realestate_type": "enebolig",
            "municipality_number": 4003,
            "lat": 59.210013830865506,
            "lon": 9.584487677348564,
            "built_year": 2016,
            "total_area": 472.8,
            "bra": 167,
            "bedrooms": 2,
            "rooms": 4,
        }
    }
    resp = client.post("/estimate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["486002054"]["estimated_price"] == int(167 * 50_000 + 472.8 * 5_000)
    assert data["486002054"]["currency"] == "NOK"
    assert data["486002054"]["model_version"] == "stub-v1"


def test_estimate_ok_business_supported(client):
    payload = {
        "99": {
            "realestate_type": "næringseiendom",
            "municipality_number": 301,
            "lat": 59.91,
            "lon": 10.75,
            "built_year": 2000,
            "total_area": 1000.0,
            "bra": 800.0,
        }
    }
    resp = client.post("/estimate", json=payload)
    assert resp.status_code == 200
    assert "99" in resp.json()


def test_estimate_requires_floor_for_leilighet(client):
    payload = {
        "1": {
            "realestate_type": "leilighet",
            "municipality_number": 301,
            "lat": 59.91,
            "lon": 10.75,
            "built_year": 2000,
            "total_area": 60.0,
            "bra": 55.0,
        }
    }
    resp = client.post("/estimate", json=payload)
    assert resp.status_code == 422


def test_estimate_total_area_must_be_gte_bra(client):
    payload = {
        "1": {
            "realestate_type": "rekkehus",
            "municipality_number": 301,
            "lat": 59.91,
            "lon": 10.75,
            "built_year": 2000,
            "total_area": 50.0,
            "bra": 55.0,
        }
    }
    resp = client.post("/estimate", json=payload)
    assert resp.status_code == 422


def test_estimate_missing_required_field_returns_422(client):
    payload = {
        "1": {
            "realestate_type": "enebolig",
            # missing municipality_number, lat/lon, built_year, areas...
        }
    }
    resp = client.post("/estimate", json=payload)
    assert resp.status_code == 422


def test_estimate_empty_payload_returns_422(client):
    resp = client.post("/estimate", json={})
    assert resp.status_code == 422
