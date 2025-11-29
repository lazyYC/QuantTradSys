import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# 1. Setup (Path, Logging, Config)
import _setup
from _setup import DEFAULT_MARKET_DB

import pandas as pd

from data_pipeline.ccxt_fetcher import (
    ensure_database,
    fetch_yearly_ohlcv,
    prune_older_rows,
    upsert_ohlcv_rows,
)
from utils.symbols import canonicalize_symbol
from utils.data_utils import dataframe_to_rows

LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill OHLCV data into SQLite store")
    parser.add_argument("symbol", help="symbol, e.g. BTC/USDT:USDT")
    parser.add_argument("timeframe", help="timeframe, e.g. 5m")
    parser.add_argument("lookback_days", type=int, help="days to backfill")
    parser.add_argument("--db", type=Path, default=DEFAULT_MARKET_DB, help="SQLite database path")
    parser.add_argument("--exchange", default="binanceusdm", help="Exchange id supported by ccxt")
    args = parser.parse_args()

    _setup.setup_logging()
    LOGGER.info("Fetching %s %s for %s days", args.symbol, args.timeframe, args.lookback_days)
    df = fetch_yearly_ohlcv(
        symbol=args.symbol,
        timeframe=args.timeframe,
        exchange_id=args.exchange,
        lookback_days=args.lookback_days,
        db_path=None,
        prune_history=False,
    )
    if df.empty:
        LOGGER.warning("No data fetched; aborting")
        return

    conn = ensure_database(args.db)
    canonical_symbol = canonicalize_symbol(args.symbol)
    rows = dataframe_to_rows(df)
    inserted = upsert_ohlcv_rows(conn, canonical_symbol, args.timeframe, rows)
    LOGGER.info("Inserted %s rows into %s for %s", inserted, args.db, canonical_symbol)

    conn.close()


if __name__ == "__main__":
    main()
