from __future__ import annotations

from typing import Tuple

STABLE_USD_EQUIVALENTS = {"USDT", "USDC", "BUSD"}


def split_symbol(symbol: str) -> Tuple[str, str]:
    """Normalize delimiters and split symbol into base/quote."""
    normalized = symbol.upper().replace(" ", "")
    if "-" in normalized:
        normalized = normalized.replace("-", "/")
    if "/" in normalized:
        base, quote = normalized.split("/", 1)
    elif len(normalized) >= 6:
        base, quote = normalized[:3], normalized[3:]
    else:
        raise ValueError(f"Cannot split symbol {symbol}")
    return base, quote


def canonicalize_symbol(symbol: str) -> str:
    """Return base/quote form where USD-stable coins are normalised to USD."""
    base, quote = split_symbol(symbol)
    if quote in STABLE_USD_EQUIVALENTS:
        quote = "USD"
    return f"{base}/{quote}"


def to_exchange_symbol(symbol: str, exchange_id: str) -> str:
    """Map canonical symbol to exchange-specific representation."""
    base, quote = split_symbol(canonicalize_symbol(symbol))
    exchange = (exchange_id or "").lower()
    if exchange in {"binance", "binanceus"}:
        if quote == "USD":
            quote = "USDT"
    return f"{base}/{quote}"


def to_alpaca_symbol(symbol: str) -> str:
    """Convert canonical symbol to Alpaca REST format."""
    base, quote = split_symbol(canonicalize_symbol(symbol))
    return f"{base}{quote}"


__all__ = ["canonicalize_symbol", "to_exchange_symbol", "to_alpaca_symbol"]
