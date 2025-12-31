
import logging
import time
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Sequence, Iterable
import ccxt
import pandas as pd

LOGGER = logging.getLogger(__name__)

def create_exchange(
    exchange_id: str = "binance", exchange_config: Optional[Dict] = None
) -> ccxt.Exchange:
    """Create a rate-limit enabled CCXT exchange instance."""
    base_config = {"enableRateLimit": True}
    merged_config = {**base_config, **(exchange_config or {})}
    try:
        exchange_cls = getattr(ccxt, exchange_id)
    except AttributeError as exc:
        raise ValueError(f"Exchange {exchange_id} is not available in ccxt") from exc
    exchange = exchange_cls(merged_config)
    LOGGER.info("Exchange %s initialized", exchange_id)
    return exchange

def timeframe_to_milliseconds(timeframe: str) -> int:
    """Convert CCXT timeframe string to milliseconds."""
    return int(ccxt.Exchange.parse_timeframe(timeframe) * 1000)

def calculate_since(end_time: datetime, lookback: timedelta) -> int:
    """Calculate start timestamp in milliseconds from end_time - lookback."""
    start_time = end_time - lookback
    return int(start_time.timestamp() * 1000)

def fetch_ohlcv_batches(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    since_ms: int,
    end_time: datetime,
    limit: int = 1000,
    pause_hook: Optional[Callable[[], None]] = None,
) -> List[Sequence[float]]:
    """Fetch OHLCV data continuously until covering the specified time range."""
    timeframe_ms = timeframe_to_milliseconds(timeframe)
    end_timestamp = int(end_time.timestamp() * 1000)
    all_rows: List[Sequence[float]] = []
    next_since = since_ms
    consecutive_empty = 0

    while next_since <= end_timestamp:
        LOGGER.debug(
            "Fetching OHLCV from %s for %s starting at %s",
            exchange.id,
            symbol,
            next_since,
        )
        try:
            batch = exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, since=next_since, limit=limit
            )
        except ccxt.NetworkError as e:
            LOGGER.warning(f"Network error during fetch: {e}")
            time.sleep(1)
            continue
            
        if not batch:
            consecutive_empty += 1
            LOGGER.warning(
                "Received empty batch (%s) for %s at since=%s",
                consecutive_empty,
                symbol,
                next_since,
            )
            if consecutive_empty >= 3:
                LOGGER.info("Stopping fetch loop due to consecutive empty batches")
                break
        else:
            consecutive_empty = 0
            all_rows.extend(batch)
            last_ts = int(batch[-1][0])
            if last_ts >= end_timestamp or len(batch) < limit:
                LOGGER.info("Fetch loop completed at %s", last_ts)
                break
            next_since = last_ts + timeframe_ms
        if pause_hook:
            pause_hook()
        elif exchange.rateLimit:
            time.sleep(exchange.rateLimit / 1000)
    return all_rows

def build_dataframe(ohlcv_rows: Iterable[Sequence[float]]) -> pd.DataFrame:
    """Convert raw OHLCV sequences to a clean pandas DataFrame."""
    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    df = pd.DataFrame(ohlcv_rows, columns=columns)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df
