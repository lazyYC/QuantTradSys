"""提供 UI 所需的資料讀取與轉換功能。"""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

from strategies.data_utils import prepare_ohlcv_frame, timeframe_to_offset

LOGGER = logging.getLogger(__name__)

MARKET_DB = Path("storage/market_data.db")
STATE_DB = Path("storage/strategy_state.db")


@dataclass(frozen=True)
class CandleQuery:
    """描述 K 線查詢條件。"""

    symbol: str
    base_timeframe: str
    start: Optional[pd.Timestamp]
    end: Optional[pd.Timestamp]
    target_timeframe: Optional[str]


@dataclass(frozen=True)
class TradeQuery:
    """描述交易紀錄查詢條件。"""

    strategy: str
    symbol: str
    timeframe: str
    start: Optional[pd.Timestamp]
    end: Optional[pd.Timestamp]
    dataset: Optional[str]


def parse_timestamp(value: Optional[str | datetime]) -> Optional[pd.Timestamp]:
    """將輸入統一轉換為 UTC 時區的 Timestamp。"""

    if value is None or value == "":
        return None
    if isinstance(value, pd.Timestamp):
        return value.tz_convert("UTC") if value.tzinfo else value.tz_localize("UTC")
    if isinstance(value, datetime):
        return pd.Timestamp(value, tz="UTC")
    try:
        parsed = pd.to_datetime(value, utc=True)
        if pd.isna(parsed):
            return None
        return parsed
    except Exception:  # noqa: BLE001
        LOGGER.warning("Failed to parse timestamp value=%s", value)
        return None


def _timestamp_to_ms(value: Optional[pd.Timestamp]) -> Optional[int]:
    """將 Timestamp 轉為毫秒整數。"""

    if value is None:
        return None
    return int(value.to_datetime64().astype("datetime64[ms]").astype(int))


def _connect(path: Path) -> sqlite3.Connection:
    """建立 SQLite 連線並盡可能採只讀模式。"""

    if not path.exists():
        raise FileNotFoundError(f"database not found: {path}")
    uri = f"file:{path.as_posix()}?mode=ro"
    try:
        return sqlite3.connect(uri, uri=True)
    except sqlite3.OperationalError:
        LOGGER.debug("SQLite readonly URI not supported, fallback to read-write mode")
        return sqlite3.connect(path)


def _timeframe_minutes(timeframe: str) -> int:
    """回傳 timeframe 對應的分鐘數。"""

    offset = timeframe_to_offset(timeframe)
    return int(offset.total_seconds() // 60)


def _to_pandas_freq(timeframe: str) -> str:
    """將 timeframe 轉成 pandas resample 可以使用的字串。"""

    value = int(timeframe[:-1])
    unit = timeframe[-1]
    mapping = {"m": "T", "h": "H", "d": "D", "w": "W"}
    if unit not in mapping:
        raise ValueError(f"不支援的 timeframe 單位: {timeframe}")
    return f"{value}{mapping[unit]}"


def _resample_ohlcv(df: pd.DataFrame, target_timeframe: str, base_timeframe: str) -> pd.DataFrame:
    """依目標 timeframe 聚合 OHLCV。"""

    base_minutes = _timeframe_minutes(base_timeframe)
    target_minutes = _timeframe_minutes(target_timeframe)
    if target_minutes < base_minutes or target_minutes % base_minutes != 0:
        raise ValueError("目標 timeframe 需為基礎 timeframe 的整數倍")
    freq = _to_pandas_freq(target_timeframe)
    indexed = df.set_index("timestamp")
    agg = indexed.resample(freq, label="right", closed="right").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    agg = agg.dropna(subset=["open", "high", "low", "close"]).reset_index()
    return agg


def fetch_candles(query: CandleQuery, market_db: Path = MARKET_DB) -> pd.DataFrame:
    """讀取指定條件的 K 線資料。"""

    conn = _connect(market_db)
    start_ms = _timestamp_to_ms(query.start)
    end_ms = _timestamp_to_ms(query.end)
    params: list = [query.symbol, query.base_timeframe]
    where_clauses: list[str] = ["symbol = ?", "timeframe = ?"]
    if start_ms is not None:
        where_clauses.append("ts >= ?")
        params.append(start_ms)
    if end_ms is not None:
        where_clauses.append("ts <= ?")
        params.append(end_ms)
    where_sql = " AND ".join(where_clauses)
    sql = (
        "SELECT ts AS timestamp, open, high, low, close, volume "
        f"FROM ohlcv WHERE {where_sql} ORDER BY ts"
    )
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    cleaned = prepare_ohlcv_frame(df, query.base_timeframe)
    if query.target_timeframe and query.target_timeframe != query.base_timeframe:
        cleaned = _resample_ohlcv(cleaned, query.target_timeframe, query.base_timeframe)
    return cleaned


def fetch_trades(query: TradeQuery, state_db: Path = STATE_DB) -> pd.DataFrame:
    """讀取策略交易紀錄。"""

    conn = _connect(state_db)
    params: list = [query.strategy, query.symbol, query.timeframe]
    where_clauses = ["strategy = ?", "symbol = ?", "timeframe = ?"]
    if query.dataset:
        where_clauses.append("dataset = ?")
        params.append(query.dataset)
    if query.start:
        where_clauses.append("exit_time >= ?")
        params.append(query.start.isoformat())
    if query.end:
        where_clauses.append("entry_time <= ?")
        params.append(query.end.isoformat())
    where_sql = " AND ".join(where_clauses)
    sql = (
        "SELECT run_id, dataset, entry_time, exit_time, side, entry_price, exit_price, return, "
        "holding_mins, entry_zscore, exit_zscore, exit_reason "
        f"FROM strategy_trades WHERE {where_sql} ORDER BY entry_time"
    )
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    if df.empty:
        return df
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    return df


def fetch_metrics(strategy: str, symbol: str, timeframe: str, state_db: Path = STATE_DB) -> pd.DataFrame:
    """讀取策略績效摘要。"""

    conn = _connect(state_db)
    sql = (
        "SELECT dataset, annualized_return, total_return, sharpe, max_drawdown, win_rate, trades, created_at "
        "FROM strategy_metrics WHERE strategy = ? AND symbol = ? AND timeframe = ? ORDER BY created_at DESC"
    )
    df = pd.read_sql_query(sql, conn, params=[strategy, symbol, timeframe])
    conn.close()
    if df.empty:
        return df
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    return df


def _safe_json(raw: Optional[str]) -> dict:
    """解析 JSON 欄位，避免錯誤影響 UI。"""

    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        LOGGER.warning("Failed to decode params_json: %s", raw)
        return {}


@lru_cache(maxsize=32)
def list_strategy_configs(state_db: Path = STATE_DB) -> list[dict[str, str]]:
    """列出策略可選項目。"""

    conn = _connect(state_db)
    rows = conn.execute(
        "SELECT strategy, symbol, timeframe, updated_at, params_json FROM strategy_params ORDER BY updated_at DESC"
    ).fetchall()
    conn.close()
    return [
        {
            "strategy": row[0],
            "symbol": row[1],
            "timeframe": row[2],
            "updated_at": row[3],
            "params": _safe_json(row[4]),
        }
        for row in rows
    ]


__all__ = [
    "CandleQuery",
    "TradeQuery",
    "fetch_candles",
    "fetch_trades",
    "fetch_metrics",
    "list_strategy_configs",
    "parse_timestamp",
]
