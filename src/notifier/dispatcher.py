import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import ccxt
import requests

from brokers import AlpacaAPIError, AlpacaCredentials, AlpacaPaperTradingClient
from config.env import DEFAULT_ENV_PATH, load_env
from utils.symbols import canonicalize_symbol, to_alpaca_symbol, to_exchange_symbol

LOGGER = logging.getLogger(__name__)

_ALPACA_CLIENT: Optional[AlpacaPaperTradingClient] = None
_BINANCE_CLIENT: Optional[ccxt.Exchange] = None


def dispatch_signal(
    action: str,
    context: Dict[str, Any],
    *,
    env_path: Optional[Path] = None,
) -> bool:
    """Dispatch strategy signal to external channels."""
    LOGGER.info("Dispatch signal=%s context=%s", action, context)
    load_env(env_path or DEFAULT_ENV_PATH)
    if action.upper() == "HOLD":
        return False

    webhook = os.getenv("DISCORD_WEBHOOK")
    if webhook:
        _send_discord(webhook, action, context)
    else:
        LOGGER.debug("DISCORD_WEBHOOK not configured, skipping Discord notification")

    client = get_alpaca_client(env_path=env_path)
    if client is None:
        LOGGER.debug("Alpaca client unavailable; trading step skipped")
        return False
    return execute_trading(client, action, context)


def execute_trading(
    client: AlpacaPaperTradingClient, action: str, context: Dict[str, Any]
) -> bool:
    symbol = _normalize_symbol(str(context.get("symbol", "")))
    action_upper = action.upper()
    success = False
    try:
        order_symbol = format_alpaca_symbol(symbol)
        if action_upper == "ENTER_LONG":
            notional = _resolve_order_notional(client)
            if notional <= 0:
                LOGGER.warning("Alpaca trading skipped: no buying power available")
                return False
            if not _is_price_within_tolerance(context, symbol):
                return False
            client.submit_order(
                symbol=order_symbol,
                side="buy",
                notional=notional,
                order_type="market",
                time_in_force="gtc",
            )
            LOGGER.info(
                "Submitted Alpaca LONG order | feed_symbol=%s order_symbol=%s notional=%.2f",
                symbol,
                order_symbol,
                notional,
            )
            success = True
        elif action_upper == "EXIT_LONG":
            try:
                client.close_position(order_symbol, side="sell")
            except AlpacaAPIError as exc:
                LOGGER.error("Failed to close LONG position for %s: %s", symbol, exc)
                return False
            LOGGER.info(
                "Closed Alpaca LONG position | feed_symbol=%s order_symbol=%s",
                symbol,
                order_symbol,
            )
            success = True
        elif action_upper == "ENTER_SHORT":
            notional = _resolve_order_notional(client)
            if notional <= 0:
                LOGGER.warning(
                    "Alpaca trading skipped: no buying power available for short"
                )
                return False
            if not _is_price_within_tolerance(context, symbol):
                return False
            client.submit_order(
                symbol=order_symbol,
                side="sell",
                notional=notional,
                order_type="market",
                time_in_force="gtc",
            )
            LOGGER.info(
                "Submitted Alpaca SHORT order | feed_symbol=%s order_symbol=%s notional=%.2f",
                symbol,
                order_symbol,
                notional,
            )
            success = True
        elif action_upper == "EXIT_SHORT":
            try:
                client.close_position(order_symbol, side="buy")
            except AlpacaAPIError as exc:
                LOGGER.error("Failed to close SHORT position for %s: %s", symbol, exc)
                return False
            LOGGER.info(
                "Closed Alpaca SHORT position | feed_symbol=%s order_symbol=%s",
                symbol,
                order_symbol,
            )
            success = True
        else:
            LOGGER.debug("Unknown trading action %s; skipping", action)
            return False
    except AlpacaAPIError as exc:
        LOGGER.error("Alpaca API error while handling %s: %s", action_upper, exc)
        return False
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Unexpected error while handling %s: %s", action_upper, exc)
        return False
    return success


def get_alpaca_client(
    *, env_path: Optional[Path] = None
) -> Optional[AlpacaPaperTradingClient]:
    """Return a cached Alpaca client after ensuring environment variables are loaded."""
    load_env(env_path or DEFAULT_ENV_PATH)
    return _get_alpaca_client()


def get_latest_price(
    _client: AlpacaPaperTradingClient, symbol: str
) -> Optional[float]:
    """Fetch latest trade price from Binance for comparison."""
    normalized = _normalize_ccxt_symbol(symbol)
    price, _ = _fetch_ccxt_price(normalized)
    return price


def _is_price_within_tolerance(context: Dict[str, Any], symbol: str) -> bool:
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

    normalized_symbol = _normalize_ccxt_symbol(symbol)
    current_price, ticker_meta = _fetch_ccxt_price(normalized_symbol)
    if current_price is None:
        LOGGER.warning("Unable to verify current price for %s; trade skipped", symbol)
        return False

    deviation = abs(current_price - signal_price_val) / signal_price_val
    if deviation <= tolerance:
        return True

    confirmed = _confirm_price_deviation(
        symbol,
        signal_price_val,
        tolerance,
        first_price=current_price,
        ticker=ticker_meta,
    )
    if confirmed is None:
        return False

    deviation, latest_price, details = confirmed
    if deviation > tolerance:
        LOGGER.warning(
            "Price deviation %.4f exceeds tolerance %.4f (signal=%.4f, latest=%.4f, details=%s); trade skipped",
            deviation,
            tolerance,
            signal_price_val,
            latest_price,
            details,
        )
        return False

    LOGGER.info(
        "Price deviation normalized after recheck (deviation=%.4f <= tolerance %.4f, details=%s)",
        deviation,
        tolerance,
        details,
    )
    return True


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
    return canonicalize_symbol(symbol or "BTC/USD")


def format_alpaca_symbol(symbol: str) -> str:
    """Format incoming symbol for Alpaca, normalising stable-coin quotes to USD."""
    canonical = canonicalize_symbol(symbol)
    base, quote = canonical.split("/", 1)
    target_quote = os.getenv("ALPACA_CRYPTO_QUOTE", "USD").upper()
    quote = target_quote or quote
    canonical_target = f"{base}/{quote}"
    strip_slash = os.getenv("ALPACA_SYMBOL_STRIP_SLASH", "true").lower() == "true"
    formatted = to_alpaca_symbol(canonical_target)
    return formatted if strip_slash else canonical_target


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

    cash_raw = account.get("cash")
    if cash_raw is None:
        LOGGER.warning("Alpaca account overview missing 'cash'; trading skipped")
        return 0.0
    try:
        cash_balance = float(cash_raw)
    except (TypeError, ValueError):
        LOGGER.error("Unable to parse Alpaca cash balance: %s", cash_raw)
        return 0.0
    if cash_balance <= 0.0:
        LOGGER.info("Alpaca account has no cash available for trading")
        return 0.0

    notional = cash_balance * order_ratio

    max_notional_raw = os.getenv("ALPACA_MAX_ORDER_NOTIONAL", "200000")
    try:
        max_notional = float(max_notional_raw)
    except ValueError:
        max_notional = 200000.0
    if max_notional <= 0:
        max_notional = None

    if max_notional is not None and notional > max_notional:
        LOGGER.debug(
            "Clamping order notional from %.2f to %.2f based on ALPACA_MAX_ORDER_NOTIONAL",
            notional,
            max_notional,
        )
        notional = max_notional

    return notional


def _fetch_ccxt_price(symbol: str) -> Tuple[Optional[float], Dict[str, Any]]:
    try:
        ticker = _get_binance_client().fetch_ticker(symbol)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to fetch Binance price for %s: %s", symbol, exc)
        return None, {}

    price_candidates = [
        ticker.get("last"),
        ticker.get("close"),
        ticker.get("info", {}).get("lastPrice"),
    ]
    price = next((float(p) for p in price_candidates if p is not None), None)
    if price is None:
        LOGGER.debug("Ticker for %s missing price fields: %s", symbol, ticker)
        return None, ticker
    return price, ticker


def _fetch_ccxt_mid_price(symbol: str) -> Optional[float]:
    try:
        order_book = _get_binance_client().fetch_order_book(symbol, limit=5)
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("Failed to fetch Binance order book for %s: %s", symbol, exc)
        return None

    bids = order_book.get("bids") or []
    asks = order_book.get("asks") or []
    best_bid = bids[0][0] if bids else None
    best_ask = asks[0][0] if asks else None
    if best_bid is None or best_ask is None:
        return None
    return float(best_bid + best_ask) / 2.0


def _confirm_price_deviation(
    symbol: str,
    signal_price: float,
    tolerance: float,
    *,
    first_price: float,
    ticker: Dict[str, Any],
) -> Optional[Tuple[float, float, Dict[str, Any]]]:
    """Re-fetch market data to confirm whether deviation is genuine."""
    ccxt_symbol = _normalize_ccxt_symbol(symbol)
    mid_price = _fetch_ccxt_mid_price(ccxt_symbol)
    second_price, second_ticker = _fetch_ccxt_price(ccxt_symbol)
    candidates = [
        ("ticker_initial", first_price),
        ("order_book_mid", mid_price),
        ("ticker_second", second_price),
    ]
    valid = [(label, price) for label, price in candidates if price is not None]
    if not valid:
        LOGGER.warning(
            "Price deviation check failed: unable to obtain confirmation quotes for %s",
            symbol,
        )
        return None

    label, price = min(
        valid,
        key=lambda item: abs(item[1] - signal_price) / signal_price,
    )
    deviation = abs(price - signal_price) / signal_price
    details = {
        "selected_source": label,
        "selected_price": price,
        "ticker_initial": ticker,
        "ticker_second": second_ticker,
        "mid_price": mid_price,
    }
    return deviation, price, details


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


def _get_binance_client() -> ccxt.Exchange:
    global _BINANCE_CLIENT
    if _BINANCE_CLIENT is None:
        _BINANCE_CLIENT = ccxt.binance({"enableRateLimit": True})
    return _BINANCE_CLIENT


def _normalize_ccxt_symbol(symbol: str) -> str:
    exchange_id = os.getenv("CCXT_PRICE_EXCHANGE", "binance")
    return to_exchange_symbol(symbol, exchange_id)


__all__ = [
    "dispatch_signal",
    "execute_trading",
    "get_alpaca_client",
    "get_latest_price",
    "format_alpaca_symbol",
]
