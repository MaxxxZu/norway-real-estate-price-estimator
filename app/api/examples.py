ESTIMATE_REQUEST_EXAMPLES = {
    "valid_batch": {
        "summary": "Valid batch: house + apartment",
        "description": "Two properties. Note: 'leilighet' requires 'floor'.",
        "value": {
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
            },
            "123456789": {
                "realestate_type": "leilighet",
                "municipality_number": 301,
                "lat": 59.913868,
                "lon": 10.752245,
                "built_year": 1998,
                "total_area": 68.0,
                "bra": 62.0,
                "floor": 3,
                "bedrooms": 1,
                "rooms": 2,
            },
        },
    },
    "invalid_missing_floor": {
        "summary": "Invalid: apartment without floor",
        "value": {
            "1": {
                "realestate_type": "leilighet",
                "municipality_number": 301,
                "lat": 59.91,
                "lon": 10.75,
                "built_year": 2000,
                "total_area": 60.0,
                "bra": 55.0,
            }
        },
    },
    "invalid_total_area_lt_bra": {
        "summary": "Invalid: total_area < bra",
        "value": {
            "1": {
                "realestate_type": "rekkehus",
                "municipality_number": 301,
                "lat": 59.91,
                "lon": 10.75,
                "built_year": 2000,
                "total_area": 50.0,
                "bra": 55.0,
            }
        },
    },
}
