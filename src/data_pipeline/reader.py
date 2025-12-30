import pandas as pd
from pathlib import Path
from typing import Optional
from persistence.market_store import MarketDataStore

def read_ohlcv(
    db_path: Path, # Deprecated, kept for signature compatibility
    symbol: str,
    timeframe: str,
    start_ts: Optional[pd.Timestamp] = None,
    end_ts: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Read OHLCV data from Market Store (PostgreSQL).
    """
    store = MarketDataStore()
    
    # helper to ensure datetime conversion if needed
    s_ts = start_ts.to_pydatetime() if start_ts else None
    e_ts = end_ts.to_pydatetime() if end_ts else None
    
    return store.load_candles(
        symbol=symbol,
        timeframe=timeframe,
        start_ts=s_ts,
        end_ts=e_ts
    )
