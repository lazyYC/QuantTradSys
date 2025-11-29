import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

LOGGER = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS strategy_params (
    strategy TEXT NOT NULL,
    study TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    params_json TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    model_path TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (strategy, study, symbol, timeframe)
);
"""


@dataclass
class StrategyRecord:
    strategy: str
    study: str
    symbol: str
    timeframe: str
    params: Dict[str, Any]
    metrics: Dict[str, Any]
    model_path: Optional[str]
    updated_at: str


def _ensure_connection(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    with conn:
        conn.executescript(SCHEMA_SQL)
        # Migration: Check if 'model_path' column exists
        cursor = conn.execute("PRAGMA table_info(strategy_params)")
        columns = {row[1] for row in cursor.fetchall()}
        if "model_path" not in columns:
            conn.execute("ALTER TABLE strategy_params ADD COLUMN model_path TEXT")
    return conn


def save_strategy_params(
    db_path: Path,
    strategy: str,
    study: str,
    symbol: str,
    timeframe: str,
    params: Dict[str, Any],
    metrics: Dict[str, Any],
    model_path: Optional[str] = None,
    *,
    stop_loss_pct: float = 0.005,
    transaction_cost: float = 0.001,
) -> StrategyRecord:
    """將最佳參數與績效指標寫入資料庫，若存在則覆蓋。"""
    conn = _ensure_connection(db_path)
    now = datetime.now(timezone.utc).isoformat()
    payload = dict(params)
    payload["stop_loss_pct"] = stop_loss_pct
    payload["transaction_cost"] = transaction_cost
    params_json = json.dumps(payload)
    metrics_json = json.dumps(metrics)
    with conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO strategy_params (
                strategy, study, symbol, timeframe, params_json, metrics_json, model_path, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (strategy, study, symbol, timeframe, params_json, metrics_json, model_path, now),
        )
    conn.close()
    LOGGER.info("Stored parameters for %s | %s | %s | %s", strategy, study, symbol, timeframe)
    return StrategyRecord(
        strategy=strategy,
        study=study,
        symbol=symbol,
        timeframe=timeframe,
        params=payload,
        metrics=metrics,
        model_path=model_path,
        updated_at=now,
    )


def load_strategy_params(
    db_path: Path,
    strategy: str,
    study: str,
    symbol: str,
    timeframe: str,
) -> Optional[StrategyRecord]:
    """讀取最新的策略參數，如果不存在則回傳 None。"""
    conn = _ensure_connection(db_path)
    cursor = conn.execute(
        """
        SELECT params_json, metrics_json, model_path, updated_at
        FROM strategy_params
        WHERE strategy = ? AND study = ? AND symbol = ? AND timeframe = ?
        """,
        (strategy, study, symbol, timeframe),
    )
    row = cursor.fetchone()
    conn.close()
    if row is None:
        return None
    params = json.loads(row[0])
    metrics = json.loads(row[1])
    model_path = row[2]
    updated_at = row[3]
    return StrategyRecord(
        strategy=strategy,
        study=study,
        symbol=symbol,
        timeframe=timeframe,
        params=params,
        metrics=metrics,
        model_path=model_path,
        updated_at=updated_at,
    )


__all__ = [
    "StrategyRecord",
    "save_strategy_params",
    "load_strategy_params",
]
