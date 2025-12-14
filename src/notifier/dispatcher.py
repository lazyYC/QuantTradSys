import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Tuple

import ccxt
import requests

from brokers import (
    AlpacaAPIError,
    AlpacaCredentials,
    AlpacaPaperTradingClient,
    BinanceAPIError,
    BinanceCredentials,
    BinanceUSDMClient,
)
from config.env import DEFAULT_ENV_PATH, load_env
from utils.symbols import canonicalize_symbol, to_alpaca_symbol, to_exchange_symbol

LOGGER = logging.getLogger(__name__)

_ALPACA_CLIENT: Optional[AlpacaPaperTradingClient] = None
_BINANCE_PRICE_CLIENT: Optional[ccxt.Exchange] = None
_BINANCE_TRADE_CLIENT: Optional[BinanceUSDMClient] = None
_DEFAULT_SYMBOL = canonicalize_symbol(os.getenv("DEFAULT_TRADING_SYMBOL", "BTC/USDT:USDT"))


class TradingBrokerAdapter(Protocol):
    """Minimal interface for brokers used by the dispatcher."""

    name: str

    def format_order_symbol(self, canonical_symbol: str) -> str:
        ...

    def submit_market_order(
        self, *, symbol: str, side: str, notional: float
    ) -> Dict[str, Any]:
        ...

    def close_position(self, symbol: str, side: Optional[str]) -> Dict[str, Any]:
        ...

    def account_overview(self) -> Dict[str, Any]:
        ...


class AlpacaBrokerAdapter:
    """Adapter exposing Alpaca client through the generic broker interface."""

    name = "alpaca"

    def __init__(self, client: AlpacaPaperTradingClient) -> None:
        self._client = client

    def format_order_symbol(self, canonical_symbol: str) -> str:
        return format_alpaca_symbol(canonical_symbol)

    def submit_market_order(
        self, *, symbol: str, side: str, notional: float
    ) -> Dict[str, Any]:
        return self._client.submit_order(
            symbol=symbol,
            side=side,
            notional=notional,
            order_type="market",
            time_in_force="gtc",
        )

    def close_position(self, symbol: str, side: Optional[str]) -> Dict[str, Any]:
        return self._client.close_position(symbol, side=side)

    def account_overview(self) -> Dict[str, Any]:
        return self._client.account_overview()


class BinanceBrokerAdapter:
    """Adapter exposing Binance USD-M client through the generic broker interface."""

    name = "binance"

    def __init__(self, client: BinanceUSDMClient) -> None:
        self._client = client

    def format_order_symbol(self, canonical_symbol: str) -> str:
        exchange_id = os.getenv("BINANCE_ORDER_EXCHANGE", "binanceusdm").lower()
        return to_exchange_symbol(canonical_symbol, exchange_id)

    def submit_market_order(
        self, *, symbol: str, side: str, notional: float
    ) -> Dict[str, Any]:
        return self._client.submit_order(
            symbol=symbol,
            side=side,
            notional=notional,
            order_type="market",
            time_in_force="GTC",
        )

    def close_position(self, symbol: str, side: Optional[str]) -> Dict[str, Any]:
        return self._client.close_position(symbol, side=side)

    def account_overview(self) -> Dict[str, Any]:
        return self._client.account_overview()


def dispatch_signal(
    action: str,
    context: Dict[str, Any],
    *,
    env_path: Optional[Path] = None,
) -> bool:
    """Dispatch strategy signal to external channels."""
    load_env(env_path or DEFAULT_ENV_PATH)
    if action.upper() == "HOLD":
        return False

    broker = get_trading_broker(env_path=env_path)

    if broker is None:
        LOGGER.warning("Trading broker unavailable; trading step skipped")
        if webhook:
            _send_discord(
                webhook,
                f"{action}_RESULT",
                {
                    "message": f"[{action.upper()}] Execution skipped: broker not available",
                },
            )
        return False

    webhook = os.getenv("DISCORD_WEBHOOK")
    if webhook:
        _send_discord(webhook, action, context)
    else:
        LOGGER.warning("DISCORD_WEBHOOK not configured, skipping Discord notification")

    trade_executed, exec_message = execute_trading(broker, action, context)

    if webhook:
        status = "SUCCESS" if trade_executed else "FAILED"
        result_context = {
            "message": f"[{action.upper()}] Execution {status}: {exec_message or 'no details'}",
        }
        if "exec_fill_price" in context:
            result_context["exec_fill_price"] = context["exec_fill_price"]
        if "signal_price" in context:
            result_context["signal_price"] = context["signal_price"]
        if "slippage_pct" in context:
            result_context["slippage_pct"] = context["slippage_pct"]
        _send_discord(webhook, f"{action}_RESULT", result_context)
        LOGGER.info(f"Dispatch signal:{action} sent to Discord successfully\n--------------------------------")

    return trade_executed


def execute_trading(
    broker: TradingBrokerAdapter, action: str, context: Dict[str, Any]
) -> Tuple[bool, str]:
    symbol = _normalize_symbol(str(context.get("symbol", "")))
    action_upper = action.upper()
    broker_label = broker.name.upper()
    success = False
    message = ""
    try:
        order_symbol = broker.format_order_symbol(symbol)
        signal_price = _extract_signal_price(context)
        exec_ref_price = _get_current_price(context, symbol, broker_name=broker.name)
        if exec_ref_price is not None:
            context["exec_ref_price"] = exec_ref_price
        if signal_price is not None and exec_ref_price is not None and signal_price != 0:
            slippage = (exec_ref_price - signal_price) / signal_price
            context["slippage_pct"] = slippage
        if action_upper == "ENTER_LONG":
            notional = _resolve_order_notional(
                broker,
                ratio=_load_order_ratio(broker.name),
                max_notional=_load_max_notional(broker.name),
                scale=context.get("scale", 1.0),
            )
            if notional <= 0:
                LOGGER.warning("%s trading skipped: no buying power available", broker_label)
                message = "Insufficient buying power"
                return False, message
            if not _is_price_within_tolerance(context, symbol, broker_name=broker.name):
                message = "Price deviation exceeded tolerance"
                return False, message
            order_resp = broker.submit_market_order(
                symbol=order_symbol, side="buy", notional=notional
            )
            _attach_fill_price(context, order_resp, broker_name=broker.name)
            LOGGER.info(
                "Submitted %s LONG order | feed_symbol=%s order_symbol=%s notional=%.2f",
                broker_label,
                symbol,
                order_symbol,
                notional,
            )
            success = True
            message = (
                f"Submitted {broker_label} LONG order | feed_symbol={symbol} "
                f"order_symbol={order_symbol} notional={notional:.2f}"
            )
        elif action_upper == "EXIT_LONG":
            scale = context.get("scale")
            if scale is not None:
                # Partial Exit (Sell)
                notional = _resolve_order_notional(
                    broker,
                    ratio=_load_order_ratio(broker.name),
                    max_notional=_load_max_notional(broker.name),
                    scale=float(scale),
                )
                if notional <= 0:
                    LOGGER.warning("%s partial exit skipped: calculated notional is zero", broker_label)
                    message = "Partial exit skipped (zero notional)"
                    return False, message
                
                order_resp = broker.submit_market_order(
                    symbol=order_symbol, side="sell", notional=notional
                )
                _attach_fill_price(context, order_resp, broker_name=broker.name)
                LOGGER.info(
                    "Submitted %s PARTIAL EXIT LONG (Sell) | feed_symbol=%s order_symbol=%s notional=%.2f scale=%.2f",
                    broker_label, symbol, order_symbol, notional, float(scale)
                )
                success = True
                message = f"Submitted {broker_label} PARTIAL EXIT LONG | scale={scale} notional={notional:.2f}"
            else:
                # Full Exit
                try:
                    order_resp = broker.close_position(order_symbol, side="sell")
                except AlpacaAPIError as exc:
                    LOGGER.error("Failed to close %s LONG position for %s: %s", broker_label, symbol, exc)
                    message = f"Failed to close LONG position: {exc}"
                    return False, message
                except BinanceAPIError as exc:
                    LOGGER.error("Failed to close %s LONG position for %s: %s", broker_label, symbol, exc)
                    message = f"Failed to close LONG position: {exc}"
                    return False, message
                _attach_fill_price(context, order_resp, broker_name=broker.name)
                LOGGER.info(
                    "Closed %s LONG position | feed_symbol=%s order_symbol=%s",
                    broker_label,
                    symbol,
                    order_symbol,
                )
                success = True
                message = (
                    f"Closed {broker_label} LONG position | feed_symbol={symbol} "
                    f"order_symbol={order_symbol}"
                )
        elif action_upper == "ENTER_SHORT":
            notional = _resolve_order_notional(
                broker,
                ratio=_load_order_ratio(broker.name),
                max_notional=_load_max_notional(broker.name),
                scale=context.get("scale", 1.0),
            )
            if notional <= 0:
                LOGGER.warning("%s trading skipped: no buying power available for short", broker_label)
                message = "Insufficient buying power for short"
                return False, message
            if not _is_price_within_tolerance(context, symbol, broker_name=broker.name):
                message = "Price deviation exceeded tolerance"
                return False, message
            order_resp = broker.submit_market_order(
                symbol=order_symbol, side="sell", notional=notional
            )
            _attach_fill_price(context, order_resp, broker_name=broker.name)
            LOGGER.info(
                "Submitted %s SHORT order | feed_symbol=%s order_symbol=%s notional=%.2f",
                broker_label,
                symbol,
                order_symbol,
                notional,
            )
            success = True
            message = (
                f"Submitted {broker_label} SHORT order | feed_symbol={symbol} "
                f"order_symbol={order_symbol} notional={notional:.2f}"
            )
        elif action_upper == "EXIT_SHORT":
            scale = context.get("scale")
            if scale is not None:
                # Partial Exit (Buy)
                notional = _resolve_order_notional(
                    broker,
                    ratio=_load_order_ratio(broker.name),
                    max_notional=_load_max_notional(broker.name),
                    scale=float(scale),
                )
                if notional <= 0:
                    LOGGER.warning("%s partial exit skipped: calculated notional is zero", broker_label)
                    message = "Partial exit skipped (zero notional)"
                    return False, message
                
                order_resp = broker.submit_market_order(
                    symbol=order_symbol, side="buy", notional=notional
                )
                _attach_fill_price(context, order_resp, broker_name=broker.name)
                LOGGER.info(
                    "Submitted %s PARTIAL EXIT SHORT (Buy) | feed_symbol=%s order_symbol=%s notional=%.2f scale=%.2f",
                    broker_label, symbol, order_symbol, notional, float(scale)
                )
                success = True
                message = f"Submitted {broker_label} PARTIAL EXIT SHORT | scale={scale} notional={notional:.2f}"
            else:
                # Full Exit
                try:
                    order_resp = broker.close_position(order_symbol, side="buy")
                except AlpacaAPIError as exc:
                    LOGGER.error("Failed to close %s SHORT position for %s: %s", broker_label, symbol, exc)
                    message = f"Failed to close SHORT position: {exc}"
                    return False, message
                except BinanceAPIError as exc:
                    LOGGER.error("Failed to close %s SHORT position for %s: %s", broker_label, symbol, exc)
                    message = f"Failed to close SHORT position: {exc}"
                    return False, message
                _attach_fill_price(context, order_resp, broker_name=broker.name)
                LOGGER.info(
                    "Closed %s SHORT position | feed_symbol=%s order_symbol=%s",
                    broker_label,
                    symbol,
                    order_symbol,
                )
                success = True
                message = (
                    f"Closed {broker_label} SHORT position | feed_symbol={symbol} "
                    f"order_symbol={order_symbol}"
                )
        elif action_upper == "NET_ZERO":
            LOGGER.info("NET_ZERO action triggered: internal stack rebalancing completed without market order.")
            success = True
            message = "Net Zero Rebalancing (No Order Submitted)"
        else:
            LOGGER.debug("Unknown trading action %s; skipping", action)
            message = f"Unknown trading action {action_upper}"
            return False, message
    except (AlpacaAPIError, BinanceAPIError) as exc:
        LOGGER.error("%s API error while handling %s: %s", broker_label, action_upper, exc)
        message = f"{broker_label} API error while handling {action_upper}: {exc}"
        return False, message
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            "Unexpected error while handling %s for %s: %s", action_upper, broker_label, exc
        )
        message = f"Unexpected error: {exc}"
        return False, message
    return success, message


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


def get_trading_broker(
    *, env_path: Optional[Path] = None
) -> Optional[TradingBrokerAdapter]:
    """Return a broker adapter based on TRADING_BROKER setting."""
    if env_path is not None:
        load_env(env_path)
    provider = (os.getenv("TRADING_BROKER") or "alpaca").strip().lower()
    if provider in {"alpaca", ""}:
        client = get_alpaca_client(env_path=env_path)
        return AlpacaBrokerAdapter(client) if client else None
    if provider in {"binance", "binanceusdm"}:
        client = _get_binance_trading_client()
        return BinanceBrokerAdapter(client) if client else None
    LOGGER.error("Unsupported TRADING_BROKER value: %s", provider)
    return None


def _extract_signal_price(context: Dict[str, Any]) -> Optional[float]:
    for key in ("signal_price", "closed_price", "price", "close"):
        val = context.get(key)
        try:
            return float(val)
        except (TypeError, ValueError):
            continue
    return None


def _get_current_price(
    context: Dict[str, Any], symbol: str, *, broker_name: str
) -> Optional[float]:
    normalized_symbol = _normalize_ccxt_symbol(symbol)
    current_price, _ = _fetch_ccxt_price(normalized_symbol)
    return current_price


def _is_price_within_tolerance(
    context: Dict[str, Any], symbol: str, *, broker_name: str
) -> bool:
    signal_price_val = _extract_signal_price(context)
    if signal_price_val is None:
        LOGGER.debug("Signal price missing or invalid; skipping price validation")
        return True

    tolerance = _load_price_tolerance(broker_name)

    normalized_symbol = _normalize_ccxt_symbol(symbol)
    current_price, ticker_meta = _fetch_ccxt_price(normalized_symbol)
    if current_price is None:
        LOGGER.warning(
            "Unable to verify current price for %s; %s trade skipped",
            symbol,
            broker_name.upper(),
        )
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
            "Price deviation %.4f exceeds tolerance %.4f (signal=%.4f, latest=%.4f, details=%s); %s trade skipped",
            deviation,
            tolerance,
            signal_price_val,
            latest_price,
            details,
            broker_name.upper(),
        )
        return False

    LOGGER.info(
        "Price deviation normalized after recheck (deviation=%.4f <= tolerance %.4f, details=%s) for %s",
        deviation,
        tolerance,
        details,
        broker_name.upper(),
    )
    return True


def _attach_fill_price(
    context: Dict[str, Any],
    order_resp: Dict[str, Any],
    *,
    broker_name: str,
) -> None:
    """Parse broker order response to extract fill price and compute slippage."""
    fill_price = _extract_fill_price(order_resp, broker_name=broker_name)
    if fill_price is None:
        LOGGER.debug("Fill price not found in %s response: %s", broker_name, order_resp)
        return
    context["exec_fill_price"] = fill_price
    signal_price = _extract_signal_price(context)
    if signal_price is not None and signal_price != 0:
        context["slippage_pct"] = (fill_price - signal_price) / signal_price


def _extract_fill_price(resp: Dict[str, Any], *, broker_name: str) -> Optional[float]:
    """Try common fields from Alpaca / Binance USD-M order responses."""
    candidates = [
        resp.get("filled_avg_price"),
        resp.get("avg_price"),
        resp.get("avgPrice"),
        resp.get("price"),
        resp.get("executedQty") and resp.get("cummulativeQuoteQty")  # Binance futures sometimes returns both; price = quote/qty
        and (
            float(resp.get("cummulativeQuoteQty"))
            / float(resp.get("executedQty") or 1)
            if float(resp.get("executedQty") or 0) > 0
            else None
        ),
    ]
    # Binance fills list
    fills = resp.get("fills") if isinstance(resp, dict) else None
    if fills and isinstance(fills, list) and fills:
        fills_price = fills[0].get("price")
        candidates.append(fills_price)
    # Binance futures user-data stream style {"avgPrice": "..."} already covered; Alpaca fill list
    alpaca_fills = resp.get("legs") or resp.get("order_fills")
    if alpaca_fills and isinstance(alpaca_fills, list):
        leg = alpaca_fills[0]
        if isinstance(leg, dict):
            candidates.append(leg.get("price"))
    for val in candidates:
        try:
            if val is None:
                continue
            price = float(val)
            if price > 0:
                return price
        except (TypeError, ValueError):
            continue
    return None


def _send_discord(webhook: str, action: str, context: Dict[str, Any]) -> None:
    message = context.get("message")
    
    if not message:
        # Auto-format for Strategy Signals
        base_msg = f"**Signal**: `{action}`"
        
        # Add Price/Target info
        price = context.get("signal_price")
        target = context.get("target_price")
        if price:
            base_msg += f"\nPrice: `{price}`"
        if target:
            base_msg += f"\nTarget: `{target:.2f}`" if isinstance(target, float) else f"\nTarget: {target}"
            
        # Add Netting/Stack Info if available (Playground specific)
        netting = context.get("netting_info")
        stacks = context.get("stack_info")
        
        if netting:
            base_msg += f"\n\n**Netting Details**:\n```\n{netting}\n```"
        if stacks:
             base_msg += f"\n**Active Stacks**:\n```\n{stacks}\n```"
             
        # Fallback for other contexts
        if not netting and not stacks:
             base_msg += f"\nContext: `{context}`"
             
        message = base_msg

    payload = {"content": message}
    try:
        response = requests.post(webhook, json=payload, timeout=10)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Failed to dispatch signal to Discord: %s", exc)
    else:
        LOGGER.info("Sent signal to Discord webhook")


def _normalize_symbol(symbol: Optional[str]) -> str:
    target = symbol or _DEFAULT_SYMBOL
    return canonicalize_symbol(target)


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


def _resolve_order_notional(
    broker: TradingBrokerAdapter, *, ratio: float, max_notional: Optional[float], scale: float = 1.0
) -> float:
    try:
        account = broker.account_overview()
    except (AlpacaAPIError, BinanceAPIError) as exc:
        LOGGER.error("Unable to fetch %s account overview: %s", broker.name.upper(), exc)
        return 0.0
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            "Unexpected error fetching %s account overview: %s", broker.name.upper(), exc
        )
        return 0.0

    if ratio <= 0.0:
        LOGGER.warning("%s order ratio is non-positive; trading skipped", broker.name.upper())
        return 0.0

    if broker.name == "alpaca":
        notional = _resolve_alpaca_notional(account, ratio)
    elif broker.name == "binance":
        notional = _resolve_binance_notional(account, ratio)
    else:
        LOGGER.error("Unsupported broker %s for notional resolution", broker.name)
        return 0.0

    LOGGER.info(
        "Resolved notional for %s: ratio=%.4f notional=%.2f (before scale/clamp)",
        broker.name.upper(), ratio, notional
    )

    if notional <= 0.0:
        return 0.0

    if scale != 1.0:
        notional *= scale
        LOGGER.info("Applied order scale %.2f -> notional %.2f", scale, notional)

    if max_notional is not None and notional > max_notional:
        LOGGER.debug(
            "Clamping order notional from %.2f to %.2f based on %s max notional",
            notional,
            max_notional,
            broker.name.upper(),
        )
        notional = max_notional
    return notional


def _resolve_alpaca_notional(account: Dict[str, Any], ratio: float) -> float:
    # Prioritize 'equity' or 'portfolio_value' for consistent sizing
    candidates = []
    for key in ("equity", "portfolio_value", "buying_power", "cash"):
        value = account.get(key)
        if value is not None:
             candidates.append((key, value))
             
    for key, val in candidates:
        try:
            amount = float(val)
        except (TypeError, ValueError):
            LOGGER.debug("Unable to parse Alpaca %s balance: %s", key, val)
            continue
        if amount > 0:
            LOGGER.info("Using Alpaca %s=%.2f for sizing", key, amount)
            return amount * ratio
            
    LOGGER.info("Alpaca account has no valid balance for trading")
    return 0.0


def _resolve_binance_notional(account: Dict[str, Any], ratio: float) -> float:
    candidates = []
    account_info = account.get("info") or {}
    
    # Prioritize Total Equity (Wallet Balance or Margin Balance)
    # totalWalletBalance: Realized Equity (Stable)
    # totalMarginBalance: Mark-to-Market Equity (Includes Unrealized PnL)
    # User requested "Total Equity", usually implies including unrealized PnL for compounding, 
    # but Wallet Balance is safer against volatility. 
    # Let's prioritize totalWalletBalance as "Total Equity".
    
    for info_key in ("totalWalletBalance", "totalMarginBalance", "availableBalance", "cashBalance"):
        value = account_info.get(info_key)
        if value is not None:
            candidates.append((info_key, value))

    usdt_entry = account.get("USDT") or {}
    for bal_key in ("total", "free"):
        value = usdt_entry.get(bal_key)
        if value is not None:
            candidates.append((f"USDT.{bal_key}", value))

    for container_key in ("total", "free"):
        container = account.get(container_key) or {}
        value = container.get("USDT")
        if value is not None:
            candidates.append((f"{container_key}.USDT", value))

    for key, raw in candidates:
        try:
            amount = float(raw)
        except (TypeError, ValueError):
            LOGGER.debug("Unable to parse Binance balance value: %s", raw)
            continue
        if amount > 0:
            LOGGER.info("Using Binance %s=%.2f for sizing", key, amount)
            return amount * ratio

    LOGGER.info("Binance account has no USDT available for trading")
    return 0.0


def _load_order_ratio(broker_name: str) -> float:
    default_ratio = 0.97
    env_keys = [
        f"{broker_name.upper()}_ORDER_RATIO",
        "ORDER_RATIO",
    ]
    if broker_name != "alpaca":
        env_keys.append("ALPACA_ORDER_RATIO")  # backward compatibility
    for key in env_keys:
        raw = os.getenv(key)
        if raw is None:
            continue
        try:
            value = max(min(float(raw), 1.0), 0.0)
            return value
        except ValueError:
            LOGGER.debug("Unable to parse %s=%s; falling back", key, raw)
    return default_ratio


def _load_max_notional(broker_name: str) -> Optional[float]:
    env_keys = [
        f"{broker_name.upper()}_MAX_ORDER_NOTIONAL",
        "MAX_ORDER_NOTIONAL",
    ]
    if broker_name != "alpaca":
        env_keys.append("ALPACA_MAX_ORDER_NOTIONAL")  # backward compatibility
    for key in env_keys:
        raw = os.getenv(key)
        if raw is None:
            continue
        try:
            value = float(raw)
        except ValueError:
            LOGGER.debug("Unable to parse %s=%s; falling back", key, raw)
            continue
        if value <= 0:
            return None
        return value
    if broker_name == "alpaca":
        return 200000.0
    return None


def _load_price_tolerance(broker_name: str) -> float:
    default_tolerance = 0.001
    env_keys = [
        f"{broker_name.upper()}_PRICE_TOLERANCE",
        "PRICE_TOLERANCE",
        "ALPACA_PRICE_TOLERANCE",  # backward compatibility
    ]
    for key in env_keys:
        raw = os.getenv(key)
        if raw is None:
            continue
        try:
            value = max(float(raw), 0.0)
            return value
        except ValueError:
            LOGGER.debug("Unable to parse %s=%s; falling back", key, raw)
    return default_tolerance


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


def _get_binance_trading_client() -> Optional[BinanceUSDMClient]:
    global _BINANCE_TRADE_CLIENT
    if _BINANCE_TRADE_CLIENT is not None:
        return _BINANCE_TRADE_CLIENT

    api_key = os.getenv("BINANCE_API_KEY") or os.getenv("BINANCE_API_KEY_ID")
    api_secret = os.getenv("BINANCE_API_SECRET") or os.getenv("BINANCE_PRIVATE_KEY")
    if not (api_key and api_secret):
        LOGGER.debug("Binance credentials not configured; trading step skipped")
        return None

    sandbox = _get_bool_env("BINANCE_SANDBOX", True)
    recv_window = _get_int_env("BINANCE_RECV_WINDOW", default=5000) or 5000
    default_symbol = os.getenv("BINANCE_DEFAULT_SYMBOL")

    try:
        client = BinanceUSDMClient(
            BinanceCredentials(api_key=api_key, api_secret=api_secret),
            sandbox=sandbox,
            recv_window=recv_window,
            default_symbol=default_symbol,
        )
    except BinanceAPIError as exc:
        LOGGER.error("Failed to initialise Binance client: %s", exc)
        return None

    _configure_binance_symbol(client)
    _BINANCE_TRADE_CLIENT = client
    LOGGER.info("Binance USD-M client initialized (sandbox=%s)", sandbox)
    return _BINANCE_TRADE_CLIENT


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
    global _BINANCE_PRICE_CLIENT
    if _BINANCE_PRICE_CLIENT is None:
        _BINANCE_PRICE_CLIENT = ccxt.binance({"enableRateLimit": True})
    return _BINANCE_PRICE_CLIENT


def _configure_binance_symbol(client: BinanceUSDMClient) -> None:
    symbol = os.getenv("BINANCE_SYMBOL") or os.getenv("BINANCE_DEFAULT_SYMBOL")
    if not symbol:
        return

    leverage = _get_int_env("BINANCE_LEVERAGE")
    margin_mode = os.getenv("BINANCE_MARGIN_MODE")
    hedged = _get_optional_bool_env("BINANCE_HEDGED_MODE")
    try:
        client.configure_symbol(
            symbol,
            leverage=leverage,
            margin_mode=margin_mode,
            hedged=hedged,
        )
    except BinanceAPIError as exc:
        LOGGER.warning("Failed to configure Binance symbol %s: %s", symbol, exc)


def _get_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_optional_bool_env(name: str) -> Optional[bool]:
    raw = os.getenv(name)
    if raw is None:
        return None
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_int_env(name: str, *, default: Optional[int] = None) -> Optional[int]:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        LOGGER.debug("Unable to parse %s=%s; falling back", name, raw)
        return default


def _normalize_ccxt_symbol(symbol: str) -> str:
    exchange_id = os.getenv("CCXT_PRICE_EXCHANGE", "binanceusdm")
    return to_exchange_symbol(symbol, exchange_id)


__all__ = [
    "AlpacaBrokerAdapter",
    "BinanceBrokerAdapter",
    "dispatch_signal",
    "execute_trading",
    "get_alpaca_client",
    "get_latest_price",
    "get_trading_broker",
    "format_alpaca_symbol",
    "TradingBrokerAdapter",
]
