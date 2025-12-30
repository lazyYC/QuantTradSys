import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import ccxt
import pandas as pd

from utils.symbols import canonicalize_symbol, to_exchange_symbol
from persistence.market_store import MarketDataStore

LOGGER = logging.getLogger(__name__)

def fetch_yearly_ohlcv(
    symbol: str,
    timeframe: str = "5m",
    exchange_id: str = "binance",
    exchange_config: Optional[Dict] = None,
    output_path: Optional[Path] = None,
    lookback_days: int = 365,
    market_store: Optional[MarketDataStore] = None,
    prune_history: bool = False,
    # Deprecated args kept for transitional compatibility or removed?
    # db_path is removed. Callers must pass store if they want persistence.
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
    configure_logging()
    canonical_symbol = canonicalize_symbol(symbol)
    exchange_symbol = to_exchange_symbol(symbol, exchange_id)
    exchange = create_exchange(exchange_id, exchange_config)
    utc_now = datetime.now(timezone.utc)
    end_timestamp = int(utc_now.timestamp() * 1000)
    lookback = timedelta(days=lookback_days)
    window_start_ms = calculate_since(utc_now, lookback)
    timeframe_ms = timeframe_to_milliseconds(timeframe)
    ongoing_open_ms = (end_timestamp // timeframe_ms) * timeframe_ms

    since_ms = window_start_ms

    # 1. Determine Start Time based on Store
    if market_store:
        last_ts = market_store.get_latest_timestamp(canonical_symbol, timeframe)
        if last_ts is not None:
             LOGGER.debug("Last stored timestamp: %s", last_ts)
             # Remove ongoing candle from store logic? 
             # Postgres Upsert handles it, but maybe we want to re-fetch the last unfinished candle.
             # Let's delete the ongoing candle to be safe or just overwrite. Store.upsert handles overwrite.
             # But if we want to re-fetch the *last closed* ?
             # Usually we fetch from last_ts + timeframe.
             # If last_ts is the "last recorded info", next fetch is last_ts + timeframe.
             
             # Logic from old fetcher:
             # Deleted where ts >= ongoing_open_ms (current open candle)
             market_store.delete_recent(canonical_symbol, timeframe, ongoing_open_ms)
             
             # Re-check last after delete?
             last_ts = market_store.get_latest_timestamp(canonical_symbol, timeframe)
             if last_ts:
                 since_ms = max(window_start_ms, last_ts + timeframe_ms)
             else:
                 since_ms = window_start_ms

    # 2. Fetch New Data
    new_rows = []
    if since_ms <= end_timestamp:
        new_rows = fetch_ohlcv_batches(
            exchange, exchange_symbol, timeframe, since_ms, utc_now
        )
        if new_rows:
            # Drop ongoing candle from memory if we don't want partials? 
            # Existing logic dropped row if < ongoing_open_ms
            new_rows = [row for row in new_rows if int(row[0]) < ongoing_open_ms]
    else:
        LOGGER.info("No new candles required for %s %s", canonical_symbol, timeframe)

    # 3. Persistence
    if market_store and new_rows:
        inserted = market_store.upsert_candles(new_rows, canonical_symbol, timeframe)
        LOGGER.info("Inserted %s rows into Store", inserted)

    # 4. Pruning
    if market_store and prune_history:
        # Not implemented in Store yet exposed as 'delete_older_than'?
        # Let's implement pruning if strict requirement.
        # But Postgres storage is cheap, maybe skip for now unless requested.
        pass

    # 5. Load Window & Backfill
    if market_store:
        # Load from store
        start_dt = datetime.fromtimestamp(window_start_ms / 1000, tz=timezone.utc)
        df = market_store.load_candles(canonical_symbol, timeframe, start_ts=start_dt)
        
        if not df.empty:
            backfilled = _backfill_missing_candles(
                market_store=market_store,
                exchange=exchange,
                exchange_symbol=exchange_symbol,
                canonical_symbol=canonical_symbol,
                timeframe=timeframe,
                df=df,
            )
            if backfilled:
                 LOGGER.info("Backfilled %s historical candles", backfilled)
                 # Reload
                 df = market_store.load_candles(canonical_symbol, timeframe, start_ts=start_dt)
    else:
        df = build_dataframe(new_rows)

    # 6. Final Filter
    if not df.empty:
        cutoff_ts = pd.to_datetime(ongoing_open_ms, unit="ms", utc=True)
        df = df[df["timestamp"] < cutoff_ts].reset_index(drop=True)

    if output_path and not df.empty:
        save_dataframe(df, output_path)

    return df


def configure_logging(level: int = logging.INFO) -> None:
    """當前尚未設定 logging handlers 時初始化預設設定。"""
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )


def create_exchange(
    exchange_id: str = "binance", exchange_config: Optional[Dict] = None
) -> ccxt.Exchange:
    """建立啟用速率限制的 CCXT 交易所實例。"""
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
    """將 CCXT 的 timeframe 字串轉換為毫秒數。"""
    return int(ccxt.Exchange.parse_timeframe(timeframe) * 1000)


def calculate_since(end_time: datetime, lookback: timedelta) -> int:
    """計算指定期間的起始毫秒時間戳。"""
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
    """持續抓取 OHLCV 資料直到涵蓋指定時間區間。"""
    timeframe_ms = timeframe_to_milliseconds(timeframe)
    end_timestamp = int(end_time.timestamp() * 1000)
    all_rows: List[Sequence[float]] = []
    next_since = since_ms
    consecutive_empty = 0

    # 透過時間戳遞增輪詢確保資料連續且無重複
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
    """將原始 OHLCV 序列轉換為整潔的 pandas DataFrame。"""
    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    df = pd.DataFrame(ohlcv_rows, columns=columns)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def save_dataframe(df: pd.DataFrame, path: Path) -> Path:
    """將 DataFrame 寫入 CSV 並回傳儲存路徑。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    LOGGER.info("Saved OHLCV data to %s", path)
    return path


def _backfill_missing_candles(
    *,
    market_store: MarketDataStore,
    exchange: ccxt.Exchange,
    exchange_symbol: str,
    canonical_symbol: str,
    timeframe: str,
    df: pd.DataFrame,
) -> int:
    """Detect gaps in stored OHLCV data and refetch missing candles."""
    timeframe_ms = timeframe_to_milliseconds(timeframe)
    windows = _find_missing_windows(df, timeframe_ms)
    if not windows:
        return 0

    total_inserted = 0
    for start_ms, end_ms in windows:
        start_iso = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
        end_iso = datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc)
        LOGGER.warning(
            "Detected gap for %s %s between %s and %s; attempting refetch",
            canonical_symbol,
            timeframe,
            start_iso,
            end_iso,
        )
        end_dt = end_iso
        rows = fetch_ohlcv_batches(
            exchange,
            exchange_symbol,
            timeframe,
            start_ms,
            end_dt,
        )
        if not rows:
            LOGGER.warning(
                "Refetch returned no candles for gap %s - %s (%s %s)",
                start_iso,
                end_iso,
                canonical_symbol,
                timeframe,
            )
            continue
        inserted = market_store.upsert_candles(rows, canonical_symbol, timeframe)
        total_inserted += inserted
    return total_inserted


def _find_missing_windows(df: pd.DataFrame, timeframe_ms: int) -> List[tuple[int, int]]:
    """Return start/end (ms) ranges for missing candles inside df."""
    if df.empty:
        return []
    timestamps = df["timestamp"].astype("int64") // 1_000_000  # to milliseconds
    windows: List[tuple[int, int]] = []
    prev = int(timestamps.iloc[0])
    for current in timestamps.iloc[1:]:
        curr_val = int(current)
        gap = curr_val - prev
        if gap > timeframe_ms:
            missing_count = gap // timeframe_ms - 1
            if missing_count > 0:
                start_missing = prev + timeframe_ms
                end_missing = curr_val - timeframe_ms
                windows.append((start_missing, end_missing))
        prev = curr_val
    return windows

__all__ = [
    "fetch_yearly_ohlcv",
    "create_exchange",
    "configure_logging",
    "build_dataframe",
    "save_dataframe",
]
