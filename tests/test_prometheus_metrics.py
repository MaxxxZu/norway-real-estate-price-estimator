def test_prometheus_metrics_endpoint(client):
    resp = client.get("/metrics/prometheus")
    assert resp.status_code == 200
    assert "http_requests_total" in resp.text
    assert "http_request_duration_seconds" in resp.text
