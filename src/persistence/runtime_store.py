import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

LOGGER = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS strategy_runtime (
    strategy TEXT NOT NULL,
    study TEXT NOT NULL DEFAULT '',
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    state_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (strategy, study, symbol, timeframe)
);
"""


@dataclass
class RuntimeRecord:
    strategy: str
    study: str
    symbol: str
    timeframe: str
    state: Dict[str, Any]
    updated_at: str


def _ensure_connection(db_path: Path) -> sqlite3.Connection:
    """確保 SQLite 資料庫存在並回傳連線。"""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    
    # Check if table exists
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='strategy_runtime'")
    table_exists = cursor.fetchone() is not None
    
    if not table_exists:
        with conn:
            conn.executescript(SCHEMA_SQL)
    else:
        # Check if study column exists
        cursor = conn.execute("PRAGMA table_info(strategy_runtime)")
        columns = [row[1] for row in cursor.fetchall()]
        if "study" not in columns:
            LOGGER.info("Migrating strategy_runtime table: adding study column")
            with conn:
                # SQLite doesn't support adding column to PK easily.
                # Since this is runtime state, we can drop and recreate or just add column and ignore PK constraint for now?
                # Actually, for runtime state, it's better to recreate or just add column.
                # But PK needs to be updated.
                # Let's just add the column for now, and rely on unique index if we had one.
                # But we have a PK.
                # Strategy: Rename table, create new, copy data.
                conn.execute("ALTER TABLE strategy_runtime RENAME TO strategy_runtime_old")
                conn.executescript(SCHEMA_SQL)
                conn.execute("""
                    INSERT INTO strategy_runtime (strategy, study, symbol, timeframe, state_json, updated_at)
                    SELECT strategy, '', symbol, timeframe, state_json, updated_at FROM strategy_runtime_old
                """)
                conn.execute("DROP TABLE strategy_runtime_old")
                
    return conn


def save_runtime_state(
    db_path: Path,
    *,
    strategy: str,
    study: str,
    symbol: str,
    timeframe: str,
    state: Dict[str, Any],
) -> RuntimeRecord:
    """將策略執行時的狀態寫入資料庫。"""
    conn = _ensure_connection(db_path)
    payload = json.dumps(state)
    now = datetime.now(timezone.utc).isoformat()
    with conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO strategy_runtime (
                strategy, study, symbol, timeframe, state_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (strategy, study, symbol, timeframe, payload, now),
        )
    conn.close()
    LOGGER.debug(
        "Saved runtime state strategy=%s study=%s symbol=%s timeframe=%s",
        strategy,
        study,
        symbol,
        timeframe,
    )
    return RuntimeRecord(
        strategy=strategy,
        study=study,
        symbol=symbol,
        timeframe=timeframe,
        state=state,
        updated_at=now,
    )


def load_runtime_state(
    db_path: Path,
    strategy: str,
    study: str,
    symbol: str,
    timeframe: str,
) -> Optional[RuntimeRecord]:
    """讀取策略先前儲存的狀態。"""
    conn = _ensure_connection(db_path)
    cursor = conn.execute(
        """
        SELECT state_json, updated_at
        FROM strategy_runtime
        WHERE strategy = ? AND study = ? AND symbol = ? AND timeframe = ?
        """,
        (strategy, study, symbol, timeframe),
    )
    row = cursor.fetchone()
    conn.close()
    if row is None:
        return None
    state_json, updated_at = row
    state = json.loads(state_json) if state_json else {}
    return RuntimeRecord(
        strategy=strategy,
        study=study,
        symbol=symbol,
        timeframe=timeframe,
        state=state,
        updated_at=updated_at,
    )


__all__ = [
    "RuntimeRecord",
    "save_runtime_state",
    "load_runtime_state",
]
