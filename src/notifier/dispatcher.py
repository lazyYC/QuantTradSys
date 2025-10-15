import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from brokers import AlpacaAPIError, AlpacaCredentials, AlpacaPaperTradingClient
from config.env import DEFAULT_ENV_PATH, load_env

LOGGER = logging.getLogger(__name__)

_ALPACA_CLIENT: Optional[AlpacaPaperTradingClient] = None


def _send_discord(webhook: str, action: str, context: Dict[str, Any]) -> None:
    message = context.get("message") or f"Signal: {action}\nContext: {context}"
    payload = {"content": message}
    try:
        response = requests.post(webhook, json=payload, timeout=10)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Failed to dispatch signal to Discord: %s", exc)
    else:
        LOGGER.info("Sent signal to Discord webhook")


def _normalize_symbol(symbol: Optional[str]) -> str:
    if not symbol:
        return "BTC/USD"
    normalized = symbol.upper().replace("USDT", "USD")
    if "-" in normalized and "/" not in normalized:
        normalized = normalized.replace("-", "/")
    if "/" not in normalized and len(normalized) > 3:
        normalized = normalized[:3] + "/" + normalized[3:]
    return normalized


def _resolve_order_notional(client: AlpacaPaperTradingClient) -> float:
    try:
        account = client.account_overview()
    except AlpacaAPIError as exc:
        LOGGER.error("Unable to fetch Alpaca account overview: %s", exc)
        return 0.0
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Unexpected error fetching Alpaca account overview: %s", exc)
        return 0.0

    ratio_raw = os.getenv("ALPACA_ORDER_RATIO", "0.97")
    try:
        order_ratio = max(min(float(ratio_raw), 1.0), 0.0)
    except ValueError:
        order_ratio = 0.97
    if order_ratio <= 0.0:
        LOGGER.warning("ALPACA_ORDER_RATIO is non-positive; trading skipped")
        return 0.0

    for key in ("buying_power", "cash"):
        value = account.get(key)
        if value is None:
            continue
        try:
            amount = float(value)
        except (TypeError, ValueError):
            continue
        if amount > 0:
            return amount * order_ratio
    return 0.0


def _fetch_latest_price(client: AlpacaPaperTradingClient, symbol: str) -> Optional[float]:
    candidates = [
        symbol.upper().replace(" ", ""),
        symbol.upper().replace("/", ""),
        symbol.upper().replace("/", "-"),
    ]
    data_url = os.getenv(
        "ALPACA_DATA_URL",
        "https://data.alpaca.markets/v1beta3/crypto/us/latest/trades",
    )
    for token in candidates:
        try:
            response = client._session.get(  # type: ignore[attr-defined]
                data_url,
                params={"symbols": token},
                timeout=5,
            )
            if response.status_code >= 400:
                LOGGER.debug("Alpaca price query failed (%s) for token %s", response.status_code, token)
                continue
            payload = response.json()
            trade = payload.get("trades", {}).get(token)
            if not trade:
                continue
            price = trade.get("p") or trade.get("price")
            if price is None:
                continue
            return float(price)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to fetch latest price for %s as %s: %s", symbol, token, exc)
            continue
    LOGGER.warning("Unable to obtain latest price for %s from Alpaca", symbol)
    return None


def get_alpaca_client(*, env_path: Optional[Path] = None) -> Optional[AlpacaPaperTradingClient]:
    """Return a cached Alpaca client after ensuring environment variables are loaded."""
    load_env(env_path or DEFAULT_ENV_PATH)
    return _get_alpaca_client()


def get_latest_price(client: AlpacaPaperTradingClient, symbol: str) -> Optional[float]:
    """Fetch latest trade price for a trading symbol."""
    normalized = _normalize_symbol(symbol)
    return _fetch_latest_price(client, normalized)


def _is_price_within_tolerance(context: Dict[str, Any], client: AlpacaPaperTradingClient, symbol: str) -> bool:
    signal_price = context.get("price") or context.get("close")
    try:
        signal_price_val = float(signal_price)
    except (TypeError, ValueError):
        LOGGER.debug("Signal price missing or invalid; skipping price validation")
        return True

    tolerance_raw = os.getenv("ALPACA_PRICE_TOLERANCE", "0.0003")
    try:
        tolerance = max(float(tolerance_raw), 0.0)
    except ValueError:
        tolerance = 0.005

    current_price = _fetch_latest_price(client, symbol)
    if current_price is None:
        LOGGER.warning("Unable to verify current price for %s; trade skipped", symbol)
        return False

    deviation = abs(current_price - signal_price_val) / signal_price_val
    if deviation > tolerance:
        LOGGER.warning(
            "Price deviation %.4f exceeds tolerance %.4f (signal=%.4f, latest=%.4f); trade skipped",
            deviation,
            tolerance,
            signal_price_val,
            current_price,
        )
        return False
    return True


def _get_alpaca_client() -> Optional[AlpacaPaperTradingClient]:
    global _ALPACA_CLIENT
    if _ALPACA_CLIENT is not None:
        return _ALPACA_CLIENT
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_SECRET")
    if not (api_key and api_secret):
        LOGGER.debug("Alpaca credentials not configured; trading step skipped")
        return None
    base_url = os.getenv("ALPACA_BASE_URL")
    creds = AlpacaCredentials(api_key=api_key, api_secret=api_secret)
    _ALPACA_CLIENT = (
        AlpacaPaperTradingClient(creds, base_url=base_url)
        if base_url
        else AlpacaPaperTradingClient(creds)
    )
    LOGGER.info("Alpaca client initialized (base_url=%s)", base_url or "default")
    return _ALPACA_CLIENT


def execute_trading(client: AlpacaPaperTradingClient, action: str, context: Dict[str, Any]) -> None:
    symbol = _normalize_symbol(str(context.get("symbol", "")))
    action_upper = action.upper()
    try:
        if action_upper == "ENTER_LONG":
            notional = _resolve_order_notional(client)
            if notional <= 0:
                LOGGER.warning("Alpaca trading skipped: no buying power available")
                return
            if not _is_price_within_tolerance(context, client, symbol):
                return
            client.submit_order(
                symbol=symbol,
                side="buy",
                notional=notional,
                order_type="market",
                time_in_force="gtc",
            )
            LOGGER.info("Submitted Alpaca LONG order | symbol=%s notional=%.2f", symbol, notional)
        elif action_upper == "EXIT_LONG":
            client.close_position(symbol, side="sell")
            LOGGER.info("Closed Alpaca LONG position | symbol=%s", symbol)
        elif action_upper == "ENTER_SHORT":
            notional = _resolve_order_notional(client)
            if notional <= 0:
                LOGGER.warning("Alpaca trading skipped: no buying power available for short")
                return
            if not _is_price_within_tolerance(context, client, symbol):
                return
            client.submit_order(
                symbol=symbol,
                side="sell",
                notional=notional,
                order_type="market",
                time_in_force="gtc",
            )
            LOGGER.info("Submitted Alpaca SHORT order | symbol=%s notional=%.2f", symbol, notional)
        elif action_upper == "EXIT_SHORT":
            client.close_position(symbol, side="buy")
            LOGGER.info("Closed Alpaca SHORT position | symbol=%s", symbol)
    except AlpacaAPIError as exc:
        LOGGER.error("Alpaca API error while handling %s: %s", action_upper, exc)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Unexpected error while handling %s: %s", action_upper, exc)


def dispatch_signal(
    action: str,
    context: Dict[str, Any],
    *,
    env_path: Optional[Path] = None,
) -> None:
    """Dispatch strategy signal to external channels."""
    LOGGER.info("Dispatch signal=%s context=%s", action, context)
    load_env(env_path or DEFAULT_ENV_PATH)
    if action.upper() == "HOLD":
        return

    webhook = os.getenv("DISCORD_WEBHOOK")
    if webhook:
        _send_discord(webhook, action, context)
    else:
        LOGGER.debug("DISCORD_WEBHOOK not configured, skipping Discord notification")

    client = get_alpaca_client(env_path=env_path)
    if client is None:
        LOGGER.debug("Alpaca client unavailable; trading step skipped")
        return
    execute_trading(client, action, context)


__all__ = ["dispatch_signal", "execute_trading", "get_alpaca_client", "get_latest_price"]
