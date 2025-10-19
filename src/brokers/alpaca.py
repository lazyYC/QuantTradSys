"""Lightweight Alpaca paper trading client for BTC execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

DEFAULT_BASE_URL = "https://paper-api.alpaca.markets"


class AlpacaAPIError(RuntimeError):
    """Raised when the Alpaca API returns an error response."""

    def __init__(
        self, status_code: int, message: str, payload: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(f"Alpaca API error ({status_code}): {message}")
        self.status_code = status_code
        self.payload = payload or {}


@dataclass
class AlpacaCredentials:
    api_key: str
    api_secret: str


class AlpacaPaperTradingClient:
    """Wrapper around the Alpaca REST API for paper trading."""

    def __init__(
        self,
        credentials: AlpacaCredentials,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 10.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "APCA-API-KEY-ID": credentials.api_key,
                "APCA-API-SECRET-KEY": credentials.api_secret,
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )

    @property
    def base_url(self) -> str:
        return self._base_url

    def submit_order(
        self,
        *,
        symbol: str,
        side: str,
        qty: Optional[float] = None,
        notional: Optional[float] = None,
        order_type: str = "market",
        time_in_force: str = "gtc",
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Place a new order on the Alpaca paper trading endpoint."""
        if qty is None and notional is None:
            raise ValueError("Either qty or notional must be provided")

        payload: Dict[str, Any] = {
            "symbol": symbol,
            "side": side.lower(),
            "type": order_type.lower(),
            "time_in_force": time_in_force.lower(),
        }
        if qty is not None:
            payload["qty"] = str(qty)
        if notional is not None:
            payload["notional"] = str(notional)
        if client_order_id:
            payload["client_order_id"] = client_order_id

        return self._post("/v2/orders", json=payload)

    def get_order(self, order_id: str) -> Dict[str, Any]:
        """Fetch a specific order."""
        return self._get(f"/v2/orders/{order_id}")

    def list_orders(self, *, status: str = "all", limit: int = 50) -> Dict[str, Any]:
        """List recent orders."""
        params = {"status": status.lower(), "limit": limit}
        return self._get("/v2/orders", params=params)

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an existing order."""
        return self._delete(f"/v2/orders/{order_id}")

    def get_position(self, symbol: str) -> Dict[str, Any]:
        """Retrieve the current position for a symbol, if any."""
        return self._get(f"/v2/positions/{symbol}")

    def close_position(
        self, symbol: str, *, side: Optional[str] = None
    ) -> Dict[str, Any]:
        """Close an open position. Optional side enforces closing long or short legs."""
        params: Dict[str, Any] = {}
        if side:
            params["side"] = side.lower()
        return self._delete(f"/v2/positions/{symbol}", params=params)

    def account_overview(self) -> Dict[str, Any]:
        """Return account status and balances."""
        return self._get("/v2/account")

    def _get(
        self, path: str, *, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        response = self._session.get(
            self._build_url(path), params=params, timeout=self._timeout
        )
        return self._handle_response(response)

    def _post(self, path: str, *, json: Dict[str, Any]) -> Dict[str, Any]:
        response = self._session.post(
            self._build_url(path), json=json, timeout=self._timeout
        )
        return self._handle_response(response)

    def _delete(
        self, path: str, *, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        response = self._session.delete(
            self._build_url(path), params=params, timeout=self._timeout
        )
        return self._handle_response(response)

    def _build_url(self, path: str) -> str:
        if not path.startswith("/"):
            path = f"/{path}"
        return f"{self._base_url}{path}"

    @staticmethod
    def _handle_response(response: requests.Response) -> Dict[str, Any]:
        if response.ok:
            if response.content:
                return response.json()
            return {}
        try:
            payload = response.json()
            message = payload.get("message") or payload.get("error", "unknown error")
        except ValueError:
            payload = {}
            message = response.text or "unknown error"
        raise AlpacaAPIError(response.status_code, message, payload=payload)
