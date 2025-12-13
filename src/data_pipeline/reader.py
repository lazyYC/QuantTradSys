import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional, List

def read_ohlcv(
    db_path: Path,
    symbol: str,
    timeframe: str,
    start_ts: Optional[pd.Timestamp] = None,
    end_ts: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Read OHLCV data from SQLite database without fetching from exchange.
    
    Args:
        db_path: Path to the SQLite database.
        symbol: Trading symbol (e.g. BTC/USDT).
        timeframe: Timeframe (e.g. 5m).
        start_ts: Optional start timestamp (inclusive).
        end_ts: Optional end timestamp (inclusive).
        
    Returns:
        DataFrame with columns [timestamp, open, high, low, close, volume].
    """
    if not db_path.exists():
        return pd.DataFrame()

    conn = sqlite3.connect(db_path)
    try:
        # Build query
        clauses = ["symbol = ?", "timeframe = ?"]
        params: List[object] = [symbol, timeframe]
        
        if start_ts is not None:
            clauses.append("ts >= ?")
            params.append(int(start_ts.timestamp() * 1000))
        if end_ts is not None:
            clauses.append("ts <= ?")
            params.append(int(end_ts.timestamp() * 1000))
            
        where_clause = " AND ".join(clauses)
        query = f"SELECT ts, open, high, low, close, volume FROM ohlcv WHERE {where_clause} ORDER BY ts"
        
        df = pd.read_sql_query(query, conn, params=params)
        
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            df = df.drop(columns=["ts"])
            
        return df
    finally:
        conn.close()
