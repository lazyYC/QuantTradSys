import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import pandas as pd

LOGGER = logging.getLogger(__name__)


def _query_dataframe(conn: sqlite3.Connection, query: str, params: Sequence[object]) -> pd.DataFrame:
    """以 pandas 讀取查詢結果並轉換時間欄位。"""
    df = pd.read_sql_query(query, conn, params=params)
    time_cols = [col for col in df.columns if col.endswith('_time') or col.endswith('_at')]
    for col in time_cols:
        df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')
    return df



def load_trades(
    db_path: Path,
    # *,
    strategy: str,
    dataset: Optional[str] = None,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    run_id: Optional[str] = None,
) -> pd.DataFrame:
    """讀取指定條件的交易紀錄。"""
    conn = _ensure_connection(db_path)
    filters = ['strategy = ?']
    params: list[object] = [strategy]
    if dataset:
        filters.append('dataset = ?')
        params.append(dataset)
    if symbol:
        filters.append('symbol = ?')
        params.append(symbol)
    if timeframe:
        filters.append('timeframe = ?')
        params.append(timeframe)
    if run_id:
        filters.append('run_id = ?')
        params.append(run_id)
    where_clause = ' AND '.join(filters)
    query = f"""
        SELECT *
        FROM strategy_trades
        WHERE {where_clause}
        ORDER BY entry_time
    """
    df = _query_dataframe(conn, query, params)
    conn.close()
    return df


def load_metrics(
    db_path: Path,
    strategy: str,
    dataset: Optional[str] = None,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    run_id: Optional[str] = None,
) -> pd.DataFrame:
    """讀取策略績效摘要供報表使用。"""
    conn = _ensure_connection(db_path)
    filters = ['strategy = ?']
    params: list[object] = [strategy]
    if dataset:
        filters.append('dataset = ?')
        params.append(dataset)
    if symbol:
        filters.append('symbol = ?')
        params.append(symbol)
    if timeframe:
        filters.append('timeframe = ?')
        params.append(timeframe)
    if run_id:
        filters.append('run_id = ?')
        params.append(run_id)
    where_clause = ' AND '.join(filters)
    query = f"""
        SELECT *
        FROM strategy_metrics
        WHERE {where_clause}
        ORDER BY created_at DESC
    """
    df = _query_dataframe(conn, query, params)
    conn.close()
    return df

TRADE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS strategy_trades (
    run_id TEXT NOT NULL,
    strategy TEXT NOT NULL,
    dataset TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    entry_time TEXT NOT NULL,
    exit_time TEXT NOT NULL,
    side TEXT NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL NOT NULL,
    return REAL NOT NULL,
    holding_mins REAL NOT NULL,
    entry_zscore REAL NOT NULL,
    exit_zscore REAL NOT NULL,
    exit_reason TEXT NOT NULL,
    PRIMARY KEY (run_id, dataset, entry_time, exit_time, symbol, timeframe)
);
"""


METRIC_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS strategy_metrics (
    run_id TEXT NOT NULL,
    strategy TEXT NOT NULL,
    dataset TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    annualized_return REAL,
    total_return REAL,
    sharpe REAL,
    max_drawdown REAL,
    win_rate REAL,
    trades INTEGER,
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, strategy, dataset, symbol, timeframe)
);
"""


def _ensure_connection(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    with conn:
        conn.executescript(TRADE_SCHEMA_SQL)
        conn.executescript(METRIC_SCHEMA_SQL)
    return conn


def save_trades(
    db_path: Path,
    *,
    strategy: str,
    dataset: str,
    symbol: str,
    timeframe: str,
    trades: pd.DataFrame,
    metrics: Mapping[str, float],
    run_id: Optional[str] = None,
) -> str:
    """將交易紀錄與對應的 summary metrics 寫入 SQLite。"""
    conn = _ensure_connection(db_path)
    run_identifier = run_id or datetime.now(timezone.utc).isoformat()
    with conn:
        if not trades.empty:
            records = [
                (
                    run_identifier,
                    strategy,
                    dataset,
                    symbol,
                    timeframe,
                    str(row["entry_time"]),
                    str(row["exit_time"]),
                    str(row["side"]),
                    float(row["entry_price"]),
                    float(row["exit_price"]),
                    float(row["return"]),
                    float(row["holding_mins"]),
                    float(row["entry_zscore"]),
                    float(row["exit_zscore"]),
                    str(row["exit_reason"]),
                )
                for _, row in trades.iterrows()
            ]
            conn.executemany(
                """
                INSERT OR REPLACE INTO strategy_trades (
                    run_id, strategy, dataset, symbol, timeframe,
                    entry_time, exit_time, side, entry_price, exit_price,
                    return, holding_mins, entry_zscore, exit_zscore, exit_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                records,
            )
        conn.execute(
            """
            INSERT OR REPLACE INTO strategy_metrics (
                run_id, strategy, dataset, symbol, timeframe,
                annualized_return, total_return, sharpe, max_drawdown,
                win_rate, trades, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_identifier,
                strategy,
                dataset,
                symbol,
                timeframe,
                float(metrics.get("annualized_return", 0.0)),
                float(metrics.get("total_return", 0.0)),
                float(metrics.get("sharpe", 0.0)),
                float(metrics.get("max_drawdown", 0.0)),
                float(metrics.get("win_rate", 0.0)),
                int(metrics.get("trades", 0)),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
    conn.close()
    LOGGER.info(
        "Stored %s trades for %s (%s) run=%s", len(trades), strategy, dataset, run_identifier
    )
    return run_identifier


def prune_strategy_trades(
    db_path: Path,
    *,
    strategy: str,
    symbol: str,
    timeframe: str,
    keep_run_id: str,
) -> int:
    """刪除舊 run 的交易紀錄，只保留指定 run_id。"""
    conn = _ensure_connection(db_path)
    with conn:
        cursor = conn.execute(
            """
            DELETE FROM strategy_trades
            WHERE strategy = ? AND symbol = ? AND timeframe = ? AND run_id <> ?
            """,
            (strategy, symbol, timeframe, keep_run_id),
        )
    conn.close()
    removed = cursor.rowcount or 0
    if removed:
        LOGGER.info("Pruned %s trades for %s | %s", removed, strategy, timeframe)
    return removed



def prune_strategy_metrics(
    db_path: Path,
    *,
    strategy: str,
    symbol: str,
    timeframe: str,
    keep_run_id: str,
) -> int:
    """刪除舊 run 的績效紀錄，只保留指定 run_id。"""
    conn = _ensure_connection(db_path)
    with conn:
        cursor = conn.execute(
            """
            DELETE FROM strategy_metrics
            WHERE strategy = ? AND symbol = ? AND timeframe = ? AND run_id <> ?
            """,
            (strategy, symbol, timeframe, keep_run_id),
        )
    conn.close()
    removed = cursor.rowcount or 0
    if removed:
        LOGGER.info("Pruned %s metrics rows for %s | %s", removed, strategy, timeframe)
    return removed


__all__ = ["save_trades", "load_trades", "load_metrics", "prune_strategy_trades", "prune_strategy_metrics"]
