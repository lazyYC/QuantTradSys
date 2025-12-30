"""
Market Data Store.
Encapsulates all OHLCV access logic using SQLAlchemy.
"""
import logging
from typing import List, Sequence, Optional, Tuple
from datetime import datetime, timezone

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
            records = data.to_dict(orient="records")
            # Transform DF records to schema format
            clean_records = []
            for r in records:
                # Ensure timestamp format
                if "ts" not in r and "timestamp" in r:
                     # timestamp is datetime64[ns, UTC]
                     ts_val = r["timestamp"]
                     if isinstance(ts_val, pd.Timestamp):
                         ts_ms = int(ts_val.timestamp() * 1000)
                         iso_str = ts_val.isoformat()
                     else:
                         # assume ms int
                         ts_ms = int(ts_val)
                         iso_str = datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc).isoformat()
                elif "ts" in r:
                    ts_ms = int(r["ts"])
                    iso_str = r.get("iso_ts") or datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc).isoformat()
                else:
                    continue  # Skip invalid

                clean_records.append({
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "ts": ts_ms,
                    "iso_ts": iso_str,
                    "open": float(r["open"]),
                    "high": float(r["high"]),
                    "low": float(r["low"]),
                    "close": float(r["close"]),
                    "volume": float(r["volume"]),
                })
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
