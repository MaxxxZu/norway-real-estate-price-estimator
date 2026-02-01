from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from app.clients.api_client import ApiClient


@dataclass(frozen=True)
class FetchConfig:
    turnover_type: int = 2
    turnover_min_price: int = 1000
    per_page: int = 100


def fetch_turnovers(
    api_client: ApiClient,
    period_start: date,
    period_end: date,
    cfg: FetchConfig,
) -> list[dict]:
    params: dict[str, Any] = {
        "start_date": period_start.isoformat(),
        "end_date": period_end.isoformat(),
        "type": cfg.turnover_type,
        "min_price": cfg.turnover_min_price,
        "page": 1,
        "per_page": cfg.per_page,
    }

    all_turnovers: list[dict] = []
    while True:
        response = api_client.get("gbk_int/turnovers/valuer_formatted", params=params)
        batch = response.get("data", [])
        all_turnovers.extend(batch)

        if len(batch) < cfg.per_page:
            break
        params["page"] += 1

    return [t.get("attributes", {}) for t in all_turnovers if isinstance(t, dict)]


def normalize_turnovers(turnovers: list[dict]) -> list[dict]:
    if not turnovers:
        return []

    normalized: list[dict] = []
    for t in turnovers:
        cadastral_ids = t.get("cadastral_unit_ids")
        if not isinstance(cadastral_ids, list) or len(cadastral_ids) != 1:
            continue

        price = t.get("price")
        if not isinstance(price, (int, float)) or price <= 0:
            continue

        raw_dt = t.get("turnover_date")
        if not isinstance(raw_dt, str):
            continue

        try:
            dt = datetime.strptime(raw_dt, "%Y-%m-%dT%H:%M:%S.%fZ")
        except ValueError:
            try:
                dt = datetime.strptime(raw_dt, "%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                continue

        normalized.append(
            {
                "turnover_date": dt.strftime("%Y-%m-%d"),
                "price": int(price),
                "cadastral_unit_ids": cadastral_ids,
            }
        )

    return normalized


def build_properties(api_client: ApiClient, cadastral_unit_ids: list[int]) -> dict[int, dict]:
    result: dict[int, dict] = {}
    if not cadastral_unit_ids:
        return result

    def chunked(items: list[int], n: int) -> list[list[int]]:
        return [items[i : i + n] for i in range(0, len(items), n)]

    for chunk in chunked(cadastral_unit_ids, 100):
        payload = {"cadastral_unit_gbk_ids": chunk}
        response = api_client.post("dc_int/units/valuer_formatted", json=payload)

        # response: { "<cadastral_id>": { full_unit, property_ids: [...] } }
        for key, data in response.items():
            try:
                cadastral_id = int(key)
            except Exception:
                continue

            if not isinstance(data, dict):
                continue

            prop_ids = data.get("property_ids", [])
            if isinstance(prop_ids, list) and len(prop_ids) == 1:
                result[cadastral_id] = data

    return result


def fetch_estimation_params(api_client: ApiClient, properties: dict[int, dict]) -> dict[str, dict]:
    property_ids: list[int] = [
        int(prop["property_ids"][0]) for prop in properties.values() if property_is_valid(prop)
    ]
    if not property_ids:
        return {}

    payload = {"ids": property_ids}
    response = api_client.post("mat_int/properties/estimation_params", json=payload)
    # response: { "<property_id>": { ...params... } }
    return response if isinstance(response, dict) else {}


def build_rows(
    turnovers: list[dict], properties: dict[int, dict], estimation_params: dict[str, dict]
) -> list[dict]:
    rows: list[dict] = []

    for t in turnovers:
        cadastral_unit_id = t["cadastral_unit_ids"][0]
        prop = properties.get(cadastral_unit_id)
        if not prop:
            continue

        prop_id = None
        prop_ids = prop.get("property_ids")
        if isinstance(prop_ids, list) and len(prop_ids) == 1:
            prop_id = prop_ids[0]
        if not isinstance(prop_id, int):
            continue

        params = estimation_params.get(str(prop_id))
        if not isinstance(params, dict) or not params:
            continue

        row = {
            "id": prop_id,
            "remote_id": prop_id,
            "price": t.get("price"),
            "turnover_date": t.get("turnover_date"),
            "cadastral_num": prop.get("full_unit"),
        }
        row.update(params)
        rows.append(row)

    return rows


def property_is_valid(property: dict) -> bool:
    if not isinstance(property, dict):
        return False

    property_ids = property.get("property_ids")
    if not isinstance(property_ids, list) or len(property_ids) != 1:
        return False

    return True
