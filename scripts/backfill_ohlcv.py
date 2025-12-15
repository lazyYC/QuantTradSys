import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# 1. Setup (Path, Logging, Config)
# Ensure src is in path for standalone execution
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config.paths import DEFAULT_MARKET_DB
from utils.logging import setup_logging

from data_pipeline.ccxt_fetcher import fetch_yearly_ohlcv

LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill OHLCV data into SQLite store")
    parser.add_argument("symbol", help="symbol, e.g. BTC/USDT:USDT")
    parser.add_argument("timeframe", help="timeframe, e.g. 5m")
    parser.add_argument("lookback_days", type=int, help="days to backfill")
    parser.add_argument("--db", type=Path, default=DEFAULT_MARKET_DB, help="SQLite database path")
    parser.add_argument("--exchange", default="binanceusdm", help="Exchange id supported by ccxt")
    args = parser.parse_args()

    setup_logging()
    LOGGER.info("Fetching %s %s for %s days", args.symbol, args.timeframe, args.lookback_days)
    
    # Use db_path to let the fetcher handle storage, gap filling, and incremental updates
    fetch_yearly_ohlcv(
        symbol=args.symbol,
        timeframe=args.timeframe,
        exchange_id=args.exchange,
        lookback_days=args.lookback_days,
        db_path=args.db,
        prune_history=False,
    )

if __name__ == "__main__":
    main()
