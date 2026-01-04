import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import ccxt
import pandas as pd

from utils.symbols import canonicalize_symbol, to_exchange_symbol
from persistence.market_store import MarketDataStore
from .client import (
    create_exchange,
    fetch_ohlcv_batches,
    build_dataframe,
    timeframe_to_milliseconds,
    calculate_since
)

LOGGER = logging.getLogger(__name__)

def fetch_yearly_ohlcv(
    symbol: str,
    timeframe: str = "5m",
    exchange_id: str = "binance",
    exchange_config: Optional[Dict] = None,
    output_path: Optional[Path] = None,
    lookback_days: int = 365,
    market_store: Optional[MarketDataStore] = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV data for the specified lookback period.
    
    If 'market_store' is provided, it performs 'Smart Sync':
      1. Check last timestamp in store.
      2. Fetch only new data.
      3. Upsert to store.
      4. Backfill any gaps in the window.
      5. Return the full window DataFrame.

    If 'market_store' is None, it fetches fresh data for the full lookback.
    """

    # configure_logging() # moved/removed
    if not logging.getLogger().handlers:
         logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    canonical_symbol = canonicalize_symbol(symbol) # e.g. BTC/USDT:USDT -> BTC/USDT
    exchange_symbol = to_exchange_symbol(symbol, exchange_id) # e.g. BTC/USDT:USDT
    exchange = create_exchange(exchange_id, exchange_config)
    utc_now = datetime.now(timezone.utc)
    now_ms = int(utc_now.timestamp() * 1000)
    
    # Calculate window
    # calculate_since returns ms timestamp for (now - delta)
    window_start_ms = calculate_since(utc_now, timedelta(days=lookback_days))
    
    timeframe_ms = timeframe_to_milliseconds(timeframe)
    ongoing_open_ms = (now_ms // timeframe_ms) * timeframe_ms
    
    # Determine the target end time (exclusive of the current open candle)
    target_end_ms = ongoing_open_ms - 1
    
    # Store-based Logic (Smart Sync)
    if market_store:
        # 1. Clean up potential partial/open candles at the tip
        market_store.delete_recent(canonical_symbol, timeframe, ongoing_open_ms)
        
        # 2. Find ALL missing intervals in the target window [window_start_ms, target_end_ms]
        # This covers both historical gaps and the "new data" at the tail.
        # It relies on SQL Window Functions which is efficient.
        gaps = market_store.find_missing_intervals(
            canonical_symbol,
            timeframe,
            start_ts=window_start_ms,
            end_ts=target_end_ms,
            interval_ms=timeframe_ms
        )
        
        total_inserted = 0
        for start_ms, end_ms in gaps:
            start_dt = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
            end_dt = datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc)
            
            LOGGER.info(
                "Fetching missing range for %s %s: %s to %s",
                canonical_symbol, timeframe, start_dt, end_dt
            )
            
            # fetch_ohlcv_batches takes (since_ms, end_datetime_utc)
            new_rows = fetch_ohlcv_batches(
                exchange, exchange_symbol, timeframe, start_ms, end_dt
            )
            
            if new_rows:
                # Filter strictly within range? batches usually handles strict since, but end might overshoot?
                # db upsert is safe on conflict, but let's filter ensuring no future data
                valid_rows = [r for r in new_rows if r[0] <= target_end_ms]
                if valid_rows:
                    count = market_store.upsert_candles(valid_rows, canonical_symbol, timeframe)
                    total_inserted += count
                    
        if total_inserted > 0:
            LOGGER.info("Total upserted rows: %s", total_inserted)
            
        # 3. Load Final Result
        start_load_dt = datetime.fromtimestamp(window_start_ms / 1000, tz=timezone.utc)
        df = market_store.load_candles(canonical_symbol, timeframe, start_ts=start_load_dt)

    else:
        # No Store: Fetch everything in memory
        new_rows = fetch_ohlcv_batches(
            exchange, exchange_symbol, timeframe, window_start_ms, utc_now
        )
        if new_rows:
            new_rows = [row for row in new_rows if int(row[0]) < ongoing_open_ms]
            
        df = build_dataframe(new_rows)

    # Final Filter
    if not df.empty:
        cutoff_ts = pd.to_datetime(ongoing_open_ms, unit="ms", utc=True)
        df = df[df["timestamp"] < cutoff_ts].reset_index(drop=True)

    if output_path and not df.empty:
        save_dataframe(df, output_path)

    return df


def save_dataframe(df: pd.DataFrame, path: Path) -> Path:
    """Save DataFrame to CSV and return path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    LOGGER.info("Saved OHLCV data to %s", path)
    return path


__all__ = [
    "fetch_yearly_ohlcv",
]
