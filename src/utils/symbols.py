from __future__ import annotations

from typing import Optional, Tuple

STABLE_USD_EQUIVALENTS = {"USDT", "USDC", "BUSD"}
QUOTE_PRIORITY = ["USDT", "USDC", "BUSD", "USD", "BTC", "ETH"]


def canonicalize_symbol(symbol: str) -> str:
    """Return a normalised symbol preserving futures suffix when present."""
    base, quote, settle = _parse_symbol(symbol)
    canonical = f"{base}/{quote}"
    if settle:
        canonical = f"{canonical}:{settle}"
    return canonical


def to_exchange_symbol(symbol: str, exchange_id: str) -> str:
    """Map canonical/on-input symbol to the representation expected by an exchange."""
    base, quote, settle = _parse_symbol(symbol)
    exchange = (exchange_id or "").lower()

    if exchange in {"binance", "binanceus"}:
        mapped_quote = _map_spot_quote(quote)
        return f"{base}/{mapped_quote}"

    if exchange == "binanceusdm":
        mapped_quote = _map_futures_quote(quote)
        settle_currency = settle or mapped_quote
        return f"{base}/{mapped_quote}:{settle_currency}"

    if exchange == "binancecoinm":
        mapped_quote = quote if quote else "USD"
        settle_currency = settle or base
        return f"{base}/{mapped_quote}:{settle_currency}"

    return f"{base}/{quote}"


def to_alpaca_symbol(symbol: str) -> str:
    """Convert canonical symbol to Alpaca REST format."""
    base, quote, _ = _parse_symbol(symbol)
    target_quote = "USD" if quote in STABLE_USD_EQUIVALENTS else quote
    return f"{base}{target_quote}"


def split_symbol(symbol: str) -> Tuple[str, str]:
    """Split into base/quote (without settlement metadata)."""
    base, quote, _ = _parse_symbol(symbol)
    return base, quote


def _parse_symbol(symbol: str) -> Tuple[str, str, Optional[str]]:
    if not symbol:
        raise ValueError("Symbol must be provided")
    normalized = symbol.upper().strip()
    if not normalized:
        raise ValueError("Symbol must be provided")
    normalized = normalized.replace(" ", "")
    settle = None
    if ":" in normalized:
        normalized, settle = normalized.split(":", 1)
    normalized = normalized.replace("\\", "/").replace("-", "/")
    if normalized.endswith("_PERP"):
        normalized = normalized[: -len("_PERP")]

    if "/" in normalized:
        base, quote = normalized.split("/", 1)
    else:
        base, quote = _split_concatenated_symbol(normalized)

    if not base or not quote:
        raise ValueError(f"Cannot split symbol {symbol}")
    return base, quote, settle


def _split_concatenated_symbol(symbol: str) -> Tuple[str, str]:
    for token in QUOTE_PRIORITY:
        if symbol.endswith(token):
            return symbol[: -len(token)], token
    if len(symbol) <= 3:
        raise ValueError(f"Cannot split symbol {symbol}")
    return symbol[:-3], symbol[-3:]


def _map_spot_quote(quote: str) -> str:
    return "USDT" if quote == "USD" else quote


def _map_futures_quote(quote: str) -> str:
    if quote == "USD":
        return "USDT"
    return quote


__all__ = ["canonicalize_symbol", "to_exchange_symbol", "to_alpaca_symbol", "split_symbol"]
