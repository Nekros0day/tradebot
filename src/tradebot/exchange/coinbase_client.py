from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlencode

import requests

from .coinbase_jwt import build_rest_jwt, format_jwt_uri
from .errors import CoinbaseAPIError


log = logging.getLogger("tradebot.coinbase")


class CoinbaseClient:
    def __init__(self, *, base_url: str, jwt_host: str, api_key_name: str, api_private_key: str, timeout_s: int = 20):
        self.base_url = base_url.rstrip("/")
        self.jwt_host = jwt_host
        self.api_key_name = api_key_name
        self.api_private_key = api_private_key
        self.timeout_s = timeout_s
        self.session = requests.Session()

    def _request(self, method: str, path: str, params: dict[str, Any] | None = None, json_body: dict[str, Any] | None = None) -> dict[str, Any]:
        path = path if path.startswith("/") else "/" + path

        jwt_uri = format_jwt_uri(method, self.jwt_host, path)  # SIGN PATH ONLY
        token = build_rest_jwt(jwt_uri, self.api_key_name, self.api_private_key)

        url = self.base_url + path

        resp = self.session.request(
            method=method.upper(),
            url=url,
            params=params,      # OK to pass params; just don't include them in JWT uri
            json=json_body,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
                **({"Content-Type": "application/json"} if json_body is not None else {}),
            },
            timeout=self.timeout_s,
        )

        if resp.status_code >= 400:
            raise CoinbaseAPIError(f"HTTP {resp.status_code}: {resp.text}")

        try:
            return resp.json()
        except Exception as e:
            raise CoinbaseAPIError(f"Failed to parse JSON: {e} | raw={resp.text[:300]}") from e

    # ---- Products / candles ----
    def get_product(self, product_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/v3/brokerage/products/{product_id}")

    def get_product_candles(self, *, product_id: str, start: int, end: int, granularity: str, limit: int = 350) -> dict[str, Any]:
        params = {
            "start": str(start),
            "end": str(end),
            "granularity": granularity,
            "limit": int(limit),
        }
        return self._request("GET", f"/api/v3/brokerage/products/{product_id}/candles", params=params)

    # ---- Accounts ----
    def list_accounts(self, limit: int = 250, cursor: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": int(limit)}
        if cursor:
            params["cursor"] = cursor
        return self._request("GET", "/api/v3/brokerage/accounts", params=params)

    def list_all_accounts(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            page = self.list_accounts(limit=250, cursor=cursor)
            out.extend(page.get("accounts", []))
            if not page.get("has_next"):
                break
            cursor = page.get("cursor")
            if not cursor:
                break
        return out

    # ---- Orders ----
    def create_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/api/v3/brokerage/orders", json_body=payload)