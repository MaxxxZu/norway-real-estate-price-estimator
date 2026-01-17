import time
from typing import Any

import httpx

from app.config import settings


class ApiClientError(RuntimeError):
    pass


class ApiClient:
    def __init__(self):
        self.base_url = settings.api_base_url.rstrip("/")
        self.headers = {"Apikey": settings.api_key}
        self._timeout = httpx.Timeout(connect=5.0, read=30.0, write=30.0, pool=5.0)
        self._client = httpx.Client(timeout=self._timeout, headers=self.headers)

    def get(self, endpoint: str, params: dict | None = None) -> dict[str, Any]:
        return self._request("GET", endpoint, params=params)

    def post(self, endpoint: str, json: dict) -> dict[str, Any]:
        return self._request("POST", endpoint, json=json)

    def _request(
        self,
        method: str,
        endpoint: str,
        params: dict | None = None,
        json: dict | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        last_exc: Exception | None = None

        for attempt in range(1, 4):
            try:
                resp = self._client.request(method, url, params=params, json=json)
                if 500 <= resp.status_code <= 599:
                    raise ApiClientError(
                        f"{method} {url} -> {resp.status_code}: {resp.text[:200]}"
                    )
                resp.raise_for_status()
                data = resp.json()
                if not isinstance(data, dict):
                    raise ApiClientError(f"Unexpected JSON type from {method} {url}: {type(data)}")
                return data
            except (httpx.RequestError, httpx.HTTPStatusError, ApiClientError) as e:
                last_exc = e
                time.sleep(0.5 * (2 ** (attempt - 1)))

        raise ApiClientError(f"Failed request after retries: {method} {url}") from last_exc
