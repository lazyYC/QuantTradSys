import logging
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import ccxt
import pandas as pd

LOGGER = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ohlcv (
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    ts INTEGER NOT NULL,
    iso_ts TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    PRIMARY KEY (symbol, timeframe, ts)
);
CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_time ON ohlcv(symbol, timeframe, ts);
"""


def configure_logging(level: int = logging.INFO) -> None:
    """當前尚未設定 logging handlers 時初始化預設設定。"""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def create_exchange(exchange_id: str = "binance", exchange_config: Optional[Dict] = None) -> ccxt.Exchange:
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
        LOGGER.debug("Fetching OHLCV from %s for %s starting at %s", exchange.id, symbol, next_since)
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=next_since, limit=limit)
        if not batch:
            consecutive_empty += 1
            LOGGER.warning("Received empty batch (%s) for %s at since=%s", consecutive_empty, symbol, next_since)
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


def ensure_database(db_path: Path) -> sqlite3.Connection:
    """建立 SQLite 資料庫連線並確保表結構存在。"""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    with conn:
        conn.executescript(SCHEMA_SQL)
        _ensure_iso_column(conn)
    return conn


def _ensure_iso_column(conn: sqlite3.Connection) -> None:
    """確保 ohlcv 表具備 ISO8601 欄位並填補缺漏資料。"""
    cursor = conn.execute("PRAGMA table_info(ohlcv)")
    columns = {row[1] for row in cursor.fetchall()}
    if "iso_ts" not in columns:
        conn.execute("ALTER TABLE ohlcv ADD COLUMN iso_ts TEXT")
    cursor = conn.execute(
        "SELECT symbol, timeframe, ts FROM ohlcv WHERE iso_ts IS NULL OR iso_ts = ''"
    )
    pending = cursor.fetchall()
    if not pending:
        return
    updates = []
    for symbol, timeframe, ts in pending:
        iso_ts = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")
        updates.append((iso_ts, symbol, timeframe, ts))
    conn.executemany(
        "UPDATE ohlcv SET iso_ts = ? WHERE symbol = ? AND timeframe = ? AND ts = ?",
        updates,
    )


def latest_timestamp(conn: sqlite3.Connection, symbol: str, timeframe: str) -> Optional[int]:
    """取得資料庫內指定商品與 timeframe 的最後時間戳。"""
    cursor = conn.execute(
        "SELECT MAX(ts) FROM ohlcv WHERE symbol = ? AND timeframe = ?",
        (symbol, timeframe),
    )
    value = cursor.fetchone()[0]
    return int(value) if value is not None else None


def upsert_ohlcv_rows(
    conn: sqlite3.Connection,
    symbol: str,
    timeframe: str,
    rows: Sequence[Sequence[float]],
) -> int:
    """將新抓取的 OHLCV 行寫入資料庫。"""
    if not rows:
        return 0
    records: List[tuple] = []
    for row in rows:
        if len(row) >= 7:
            ts_ms = int(row[0])
            iso_ts = str(row[1]) or datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")
            open_, high, low, close, volume = row[2:7]
        elif len(row) == 6:
            ts_ms = int(row[0])
            iso_ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")
            open_, high, low, close, volume = row[1:6]
        else:
            raise ValueError("Unexpected OHLCV row format; cannot upsert")
        records.append(
            (
                symbol,
                timeframe,
                ts_ms,
                iso_ts,
                float(open_),
                float(high),
                float(low),
                float(close),
                float(volume),
            )
        )
    with conn:
        conn.executemany(
            """
            INSERT OR IGNORE INTO ohlcv (
                symbol, timeframe, ts, iso_ts, open, high, low, close, volume
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )
    return len(records)


def prune_older_rows(
    conn: sqlite3.Connection,
    symbol: str,
    timeframe: str,
    keep_from_ms: int,
) -> int:
    """清理超過觀察窗口的舊資料以節省儲存。"""
    with conn:
        cursor = conn.execute(
            "DELETE FROM ohlcv WHERE symbol = ? AND timeframe = ? AND ts < ?",
            (symbol, timeframe, keep_from_ms),
        )
    return cursor.rowcount


def load_ohlcv_window(
    conn: sqlite3.Connection,
    symbol: str,
    timeframe: str,
    start_ms: int,
) -> pd.DataFrame:
    """從資料庫載入指定時間範圍內的資料並轉為 DataFrame。"""
    query = (
        "SELECT ts AS timestamp, open, high, low, close, volume "
        "FROM ohlcv WHERE symbol = ? AND timeframe = ? AND ts >= ? ORDER BY ts"
    )
    df = pd.read_sql_query(query, conn, params=(symbol, timeframe, start_ms))
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def fetch_yearly_ohlcv(
    symbol: str,
    timeframe: str = "5m",
    exchange_id: str = "binance",
    exchange_config: Optional[Dict] = None,
    output_path: Optional[Path] = None,
    lookback_days: int = 365,
    db_path: Optional[Path] = Path("storage/market_data.db"),
    prune_history: bool = False,
) -> pd.DataFrame:
    """抓取約一年的 OHLCV 資料，並支援增量更新與 SQLite 儲存。"""
    configure_logging()
    exchange = create_exchange(exchange_id, exchange_config)
    utc_now = datetime.now(timezone.utc)
    end_timestamp = int(utc_now.timestamp() * 1000)
    lookback = timedelta(days=lookback_days)
    window_start_ms = calculate_since(utc_now, lookback)
    timeframe_ms = timeframe_to_milliseconds(timeframe)

    conn: Optional[sqlite3.Connection] = None
    if db_path is not None:
        conn = ensure_database(db_path)
        last_ts = latest_timestamp(conn, symbol, timeframe)
        LOGGER.debug("Last stored timestamp: %s", last_ts)
        since_ms = max(window_start_ms, (last_ts + timeframe_ms) if last_ts is not None else window_start_ms)
    else:
        since_ms = window_start_ms

    if since_ms <= end_timestamp:
        new_rows = fetch_ohlcv_batches(exchange, symbol, timeframe, since_ms, utc_now)
    else:
        LOGGER.info("No new candles required for %s %s", symbol, timeframe)
        new_rows = []

    if conn is not None:
        inserted = upsert_ohlcv_rows(conn, symbol, timeframe, new_rows)
        LOGGER.info("Inserted %s rows into %s", inserted, db_path)
        if prune_history:
            pruned = prune_older_rows(conn, symbol, timeframe, window_start_ms)
            if pruned:
                LOGGER.debug("Pruned %s obsolete rows", pruned)
        df = load_ohlcv_window(conn, symbol, timeframe, window_start_ms)
        conn.close()
    else:
        df = build_dataframe(new_rows)

    if output_path and not df.empty:
        save_dataframe(df, output_path)
    return df


__all__ = [
    "fetch_yearly_ohlcv",
    "create_exchange",
    "configure_logging",
    "build_dataframe",
    "save_dataframe",
    "ensure_database",
    "load_ohlcv_window",
]
