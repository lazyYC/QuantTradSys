"""Lightweight Binance WebSocket utilities for kline subscriptions."""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import websocket

LOGGER = logging.getLogger(__name__)

_BINANCE_STREAM_URL = "wss://stream.binance.com:9443/ws"


def _format_stream_symbol(symbol: str) -> str:
    """Convert symbols like 'BTC/USDT' to 'btcusdt'."""
    return symbol.replace("/", "").lower()


def _build_stream_url(symbol: str, timeframe: str) -> str:
    stream = f"{_format_stream_symbol(symbol)}@kline_{timeframe}"
    return f"{_BINANCE_STREAM_URL}/{stream}"


ClosedKlineCallback = Callable[[dict], None]


@dataclass
class BinanceKlineSubscriber:
    """Simple reconnection-aware subscriber for Binance kline streams."""

    symbol: str
    timeframe: str
    on_closed_kline: ClosedKlineCallback
    idle_sleep: float = 1.0
    reconnect_delay: float = 5.0

    def __post_init__(self) -> None:
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._ws: Optional[websocket.WebSocketApp] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            LOGGER.warning("Subscriber already running")
            return
        self._running.set()
        self._thread = threading.Thread(target=self._run_loop, name="BinanceKlineSubscriber", daemon=True)
        self._thread.start()
        LOGGER.info("Started Binance kline subscriber for %s %s", self.symbol, self.timeframe)

    def stop(self) -> None:
        self._running.clear()
        if self._ws:
            try:
                self._ws.close()
            except Exception:  # pragma: no cover - defensive
                LOGGER.debug("Error closing websocket", exc_info=True)
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        LOGGER.info("Stopped Binance kline subscriber for %s %s", self.symbol, self.timeframe)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _run_loop(self) -> None:
        url = _build_stream_url(self.symbol, self.timeframe)
        while self._running.is_set():
            LOGGER.info("Connecting to %s", url)
            self._ws = websocket.WebSocketApp(
                url,
                on_message=self._handle_message,
                on_error=self._handle_error,
                on_close=self._handle_close,
            )
            try:
                self._ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception:  # pragma: no cover - defensive
                LOGGER.exception("WebSocket run_forever raised an exception")

            if not self._running.is_set():
                break
            LOGGER.info("Reconnecting in %.1f seconds", self.reconnect_delay)
            time.sleep(self.reconnect_delay)

    # ------------------------------------------------------------------ #
    # WebSocket callbacks
    # ------------------------------------------------------------------ #

    def _handle_message(self, _ws: websocket.WebSocketApp, message: str) -> None:
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            LOGGER.debug("Dropped malformed message: %s", message)
            return
        kline = payload.get("k")
        if not kline:
            return
        if kline.get("x"):
            try:
                self.on_closed_kline(kline)
            except Exception:
                LOGGER.exception("Error while processing closed kline callback")

    def _handle_error(self, _ws: websocket.WebSocketApp, error: Exception) -> None:
        LOGGER.warning("WebSocket error: %s", error)

    def _handle_close(self, _ws: websocket.WebSocketApp, status_code: int, msg: str) -> None:
        LOGGER.info("WebSocket closed (%s) %s", status_code, msg)


__all__ = ["BinanceKlineSubscriber"]
