import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from utils.logging import setup_logging

from persistence.market_store import MarketDataStore
from data_pipeline.ccxt_fetcher import fetch_yearly_ohlcv

LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill OHLCV data into Market Store")
    parser.add_argument("symbol", help="symbol, e.g. BTC/USDT:USDT")
    parser.add_argument("timeframe", help="timeframe, e.g. 5m")
    parser.add_argument("lookback_days", type=int, help="days to backfill")
    parser.add_argument("--exchange", default="binanceusdm", help="Exchange id supported by ccxt")
    args = parser.parse_args()

    setup_logging()
    LOGGER.info("Fetching %s %s for %s days", args.symbol, args.timeframe, args.lookback_days)
    
    store = MarketDataStore()
    
    fetch_yearly_ohlcv(
        symbol=args.symbol,
        timeframe=args.timeframe,
        exchange_id=args.exchange,
        lookback_days=args.lookback_days,
        market_store=store,
    )

if __name__ == "__main__":
    main()
