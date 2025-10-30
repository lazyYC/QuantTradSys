"""Binance USD-M futures broker utilities built on top of ccxt."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import ccxt
from ccxt.base.errors import BaseError as CCXTBaseError
from ccxt.base.errors import ExchangeError, InvalidNonce

LOGGER = logging.getLogger(__name__)


class BinanceAPIError(RuntimeError):
    """Raised when Binance API returns an unexpected response."""

    def __init__(self, message: str, *, payload: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(f"Binance API error: {message}")
        self.payload = payload or {}


@dataclass
class BinanceCredentials:
    api_key: str
    api_secret: str


class BinanceUSDMClient:
    """Lightweight wrapper for Binance USD-M futures via ccxt."""

    def __init__(
        self,
        credentials: BinanceCredentials,
        *,
        sandbox: bool = True,
        timeout: float = 10.0,
        adjust_time_diff: bool = True,
        recv_window: int = 5000,
        default_symbol: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        exchange_kwargs: Dict[str, Any] = {
            "apiKey": credentials.api_key,
            "secret": credentials.api_secret,
            "enableRateLimit": True,
            "timeout": int(timeout * 1000),
            "options": {"defaultType": "future"},
        }
        if options:
            exchange_kwargs["options"].update(options)

        self._exchange = ccxt.binanceusdm(exchange_kwargs)
        if sandbox:
            self._exchange.set_sandbox_mode(True)

        self._adjust_time_diff = adjust_time_diff
        self._recv_window = recv_window
        self._default_symbol = default_symbol

        if self._adjust_time_diff:
            self._sync_time(force=False)

    @property
    def exchange(self) -> ccxt.Exchange:
        return self._exchange

    def configure_symbol(
        self,
        symbol: str,
        *,
        leverage: Optional[int] = None,
        margin_mode: Optional[str] = None,
        hedged: Optional[bool] = None,
    ) -> None:
        if hedged is not None:
            self._call_with_time_sync(
                self._exchange.set_position_mode, hedged=hedged, symbol=symbol
            )
        if margin_mode is not None:
            mode = margin_mode.lower()
            self._call_with_time_sync(
                self._exchange.set_margin_mode, marginMode=mode, symbol=symbol
            )
        if leverage is not None:
            self._call_with_time_sync(self._exchange.set_leverage, leverage, symbol)

    def account_overview(self) -> Dict[str, Any]:
        return self._call_with_time_sync(
            self._exchange.fetch_balance, params={"type": "future"}
        )

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        positions = self._call_with_time_sync(
            self._exchange.fetch_positions, symbols=[symbol]
        )
        for position in positions:
            if position.get("symbol") == symbol:
                return position
        return None

    def close_position(
        self,
        symbol: str,
        *,
        side: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        position = self.get_position(symbol)
        if not position:
            return {}

        raw_amt = position.get("info", {}).get("positionAmt")
        if raw_amt is None:
            return {}
        amount = abs(float(raw_amt))
        if amount == 0:
            return {}

        inferred_side = "sell" if float(raw_amt) > 0 else "buy"
        order_side = side.lower() if side else inferred_side
        return self.submit_order(
            symbol=symbol,
            side=order_side,
            qty=amount,
            order_type="market",
            reduce_only=True,
            params=params,
        )

    def submit_order(
        self,
        *,
        symbol: str,
        side: str,
        qty: Optional[float] = None,
        notional: Optional[float] = None,
        order_type: str = "market",
        time_in_force: str = "GTC",
        price: Optional[float] = None,
        reduce_only: bool = False,
        client_order_id: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if qty is None and notional is None:
            raise ValueError("Either qty or notional must be provided")

        order_params: Dict[str, Any] = {
            "recvWindow": self._recv_window,
            "timeInForce": time_in_force.upper(),
        }
        if reduce_only:
            order_params["reduceOnly"] = True
        if params:
            order_params.update(params)
        if client_order_id:
            order_params["newClientOrderId"] = client_order_id

        market_symbol = symbol
        amount = qty
        if amount is None:
            amount = self._resolve_amount_from_notional(
                market_symbol, notional=notional, price=price
            )
        precision_amount = self._exchange.amount_to_precision(market_symbol, amount)
        amount_value = float(precision_amount)
        if amount_value <= 0:
            raise BinanceAPIError("Order amount resolves to zero")

        order_price = price
        order_type_lower = order_type.lower()
        if order_type_lower == "limit":
            if order_price is None:
                raise ValueError("Limit orders require price")
            order_price = float(self._exchange.price_to_precision(symbol, order_price))
        else:
            order_price = None

        try:
            return self._call_with_time_sync(
                self._exchange.create_order,
                symbol,
                order_type_lower,
                side.lower(),
                amount_value,
                order_price,
                order_params,
            )
        except CCXTBaseError as exc:
            raise BinanceAPIError(str(exc)) from exc

    def cancel_order(
        self, order_id: str, *, symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        resolved_symbol = symbol or self._require_symbol()
        return self._call_with_time_sync(
            self._exchange.cancel_order,
            order_id,
            resolved_symbol,
            params={"recvWindow": self._recv_window},
        )

    def list_orders(
        self,
        *,
        symbol: Optional[str] = None,
        since: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Any:
        resolved_symbol = symbol or self._require_symbol()
        return self._call_with_time_sync(
            self._exchange.fetch_orders,
            resolved_symbol,
            since,
            limit,
            {"recvWindow": self._recv_window},
        )

    def get_order(
        self, order_id: str, *, symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        resolved_symbol = symbol or self._require_symbol()
        return self._call_with_time_sync(
            self._exchange.fetch_order,
            order_id,
            resolved_symbol,
            params={"recvWindow": self._recv_window},
        )

    def _resolve_amount_from_notional(
        self,
        symbol: str,
        *,
        notional: Optional[float],
        price: Optional[float],
    ) -> float:
        if notional is None:
            raise ValueError("notional must be provided")

        market_price = price
        if market_price is None:
            ticker = self._call_with_time_sync(self._exchange.fetch_ticker, symbol)
            market_price = next(
                (
                    float(candidate)
                    for candidate in (
                        ticker.get("last"),
                        ticker.get("close"),
                        ticker.get("info", {}).get("lastPrice"),
                    )
                    if candidate is not None
                ),
                None,
            )
        if market_price is None or market_price <= 0:
            raise BinanceAPIError("Unable to determine market price for notional order")
        return float(notional) / float(market_price)

    def _call_with_time_sync(self, func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (InvalidNonce, ExchangeError) as exc:
            if not self._adjust_time_diff or not self._is_timestamp_error(exc):
                raise BinanceAPIError(str(exc)) from exc
            LOGGER.warning("Timestamp mismatch detected, syncing server time...")
            self._sync_time(force=True)
            return func(*args, **kwargs)
        except CCXTBaseError as exc:
            raise BinanceAPIError(str(exc)) from exc

    def _sync_time(self, *, force: bool) -> None:
        try:
            diff = self._exchange.load_time_difference()
            LOGGER.info("Binance time difference synced: %.0f ms", diff)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to sync Binance time difference: %s", exc)
            if force:
                raise BinanceAPIError("Unable to sync Binance server time") from exc

    @staticmethod
    def _is_timestamp_error(exc: Exception) -> bool:
        message = str(exc)
        return "-1021" in message or "Timestamp for this request" in message

    def _require_symbol(self) -> str:
        if not self._default_symbol:
            raise ValueError("Symbol must be provided for this operation")
        return self._default_symbol


__all__ = [
    "BinanceAPIError",
    "BinanceCredentials",
    "BinanceUSDMClient",
]
