"""
Market Data Store.
Encapsulates all OHLCV access logic using SQLAlchemy.
"""
import logging
from typing import List, Sequence, Optional, Tuple
from datetime import datetime, timezone
import numpy as np

import pandas as pd
from sqlalchemy import select, delete, func
from sqlalchemy.dialects.postgresql import insert

from database.connection import get_session, get_engine
from database.schema import OHLCV

LOGGER = logging.getLogger(__name__)


class MarketDataStore:
    def __init__(self):
        # Ensure table exists? 
        # Usually handled by migration script or startup hook.
        pass

    def get_latest_timestamp(self, symbol: str, timeframe: str) -> Optional[int]:
        """Get the last stored timestamp (ms) for a symbol."""
        with get_session() as session:
            stmt = select(func.max(OHLCV.ts)).where(
                OHLCV.symbol == symbol,
                OHLCV.timeframe == timeframe
            )
            result = session.execute(stmt).scalar()
            return int(result) if result is not None else None

    def upsert_candles(self, data: List[tuple] | pd.DataFrame, symbol: str, timeframe: str) -> int:
        """
        Upsert candles into the database.
        Accepts list of tuples or DataFrame.
        Tuples expected: (ts_ms, iso_str, open, high, low, close, volume)
        """
        if isinstance(data, pd.DataFrame):
            if data.empty:
                return 0
                
            # Vectorized processing
            df = data.copy()
            
            # 1. Ensure 'timestamp' is handled (ccxt_fetcher uses 'timestamp')
            # If 'timestamp' exists and is datetime
            if "timestamp" in df.columns:
                # Ensure UTC
                if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                    # If already tz-aware, convert to UTC, else localize
                    if df["timestamp"].dt.tz is None:
                        df["timestamp"] = df["timestamp"].dt.tz_localize(timezone.utc)
                    else:
                        df["timestamp"] = df["timestamp"].dt.tz_convert(timezone.utc)
                        
                df["ts"] = df["timestamp"].astype(np.int64) // 10**6
                df["iso_ts"] = df["timestamp"].apply(lambda x: x.isoformat())
            
            # If input has 'ts' but no 'timestamp' or 'iso_ts' (e.g. from some other source)
            elif "ts" in df.columns and "iso_ts" not in df.columns:
                df["iso_ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True).apply(lambda x: x.isoformat())
            
            # 2. Add metadata
            df["symbol"] = symbol
            df["timeframe"] = timeframe
            
            # 3. Select columns matching DB schema
            # Schema: symbol, timeframe, ts, iso_ts, open, high, low, close, volume
            required_cols = ["symbol", "timeframe", "ts", "iso_ts", "open", "high", "low", "close", "volume"]
            
            # Ensure float type for OHLCV
            for col in ["open", "high", "low", "close", "volume"]:
                 df[col] = df[col].astype(float)
                 
            # Convert to dict records
            clean_records = df[required_cols].to_dict(orient="records")

        else:
             # Assume list of tuples from legacy fetcher
             # (ts, iso, o, h, l, c, v)
             if not data:
                 return 0
             clean_records = []
             for row in data:
                 # Helper to robustly handle the tuple (can be len 6 or 7)
                 if len(row) >= 7:
                     ts_ms = int(row[0])
                     iso_ts = str(row[1])
                     o, h, l, c, v = row[2:7]
                 elif len(row) == 6:
                     ts_ms = int(row[0])
                     iso_ts = None # Let DB default or fill
                     o, h, l, c, v = row[1:6]
                 else:
                     continue
                 
                 clean_records.append({
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "ts": ts_ms,
                    "iso_ts": iso_ts,
                    "open": float(o),
                    "high": float(h),
                    "low": float(l),
                    "close": float(c),
                    "volume": float(v),
                 })

        if not clean_records:
            return 0

        # Postgres Upsert
        with get_session() as session:
            stmt = insert(OHLCV).values(clean_records)
            stmt = stmt.on_conflict_do_update(
                index_elements=["symbol", "timeframe", "ts"],
                set_={
                    "open": stmt.excluded.open,
                    "high": stmt.excluded.high,
                    "low": stmt.excluded.low,
                    "close": stmt.excluded.close,
                    "volume": stmt.excluded.volume,
                    "iso_ts": stmt.excluded.iso_ts,
                }
            )
            result = session.execute(stmt)
            return result.rowcount

    def load_candles(
        self,
        symbol: str, 
        timeframe: str, 
        start_ts: Optional[datetime] = None, 
        end_ts: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load candles as a pandas DataFrame.
        start_ts / end_ts are datetime objects (UTC).
        """
        stmt = select(OHLCV).where(
            OHLCV.symbol == symbol,
            OHLCV.timeframe == timeframe
        )
        
        if start_ts:
            start_ms = int(start_ts.timestamp() * 1000)
            stmt = stmt.where(OHLCV.ts >= start_ms)
        if end_ts:
            end_ms = int(end_ts.timestamp() * 1000)
            stmt = stmt.where(OHLCV.ts <= end_ms)
            
        stmt = stmt.order_by(OHLCV.ts.asc())
        
        if limit:
            stmt = stmt.limit(limit)

        with get_engine().connect() as conn:
            df = pd.read_sql_query(stmt, conn)
            
        if df.empty:
            return pd.DataFrame()
        
        # Format Timestamp
        df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        # Sort cols
        return df[["timestamp", "open", "high", "low", "close", "volume"]]

    def delete_recent(self, symbol: str, timeframe: str, since_ms: int) -> int:
        """Delete rows since a specific timestamp (inclusive)."""
        with get_session() as session:
            stmt = delete(OHLCV).where(
                OHLCV.symbol == symbol,
                OHLCV.timeframe == timeframe,
                OHLCV.ts >= since_ms
            )
            result = session.execute(stmt)
            return result.rowcount

    def delete_tail(self, symbol: str, timeframe: str, limit: int) -> int:
        """Delete the most recent 'limit' candles."""
        if limit <= 0:
            return 0
            
        with get_session() as session:
            # Subquery to identify the timestamps to delete
            subquery = select(OHLCV.ts).where(
                OHLCV.symbol == symbol,
                OHLCV.timeframe == timeframe
            ).order_by(OHLCV.ts.desc()).limit(limit)
            
            stmt = delete(OHLCV).where(
                OHLCV.symbol == symbol,
                OHLCV.timeframe == timeframe,
                OHLCV.ts.in_(subquery)
            )
            result = session.execute(stmt)
            return result.rowcount
