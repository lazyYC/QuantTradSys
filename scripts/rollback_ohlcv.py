import argparse
from pathlib import Path
from datetime import datetime
import sys

# 1. Setup Path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from persistence.market_store import MarketDataStore
from utils.logging import setup_logging
from utils.symbols import canonicalize_symbol
from utils.formatting import format_ts

# Setup Logging
setup_logging()

MILLIS_PER_DAY = 86_400_000

def main() -> None:
    parser = argparse.ArgumentParser(description="Rollback recent OHLCV data from the Market Store.")
    parser.add_argument("--symbol", required=True, help="Trading symbol, e.g. BTC/USDT")
    parser.add_argument("--timeframe", required=True, help="Timeframe, e.g. 5m")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--days", type=float, help="Rollback the most recent N days of candles")
    group.add_argument("--limit", type=int, help="Rollback the most recent N candles")
    parser.add_argument("--dry-run", action="store_true", help="Just print what would be done (Not fully supported by Store yet, prints expected action)")
    args = parser.parse_args()

    store = MarketDataStore()
    symbol = canonicalize_symbol(args.symbol)
    
    if args.dry_run:
        print(f"[DRY-RUN] Would delete data for {symbol} {args.timeframe}")
        if args.days:
             print(f"  Mode: Recent {args.days} days")
        else:
             print(f"  Mode: Recent {args.limit} candles")
        return

    if args.days is not None:
        rollback_recent_days(store, symbol, args.timeframe, args.days)
    else:
        rollback_recent_limit(store, symbol, args.timeframe, args.limit)


def rollback_recent_days(
    store: MarketDataStore, symbol: str, timeframe: str, days: float
) -> int:
    if days <= 0:
        raise ValueError("days must be positive")
    
    # 1. Get max timestamp
    max_ts = store.get_latest_timestamp(symbol, timeframe)
    if max_ts is None:
        print("No data found for the specified symbol/timeframe.")
        return 0
        
    cutoff = int(max_ts - days * MILLIS_PER_DAY)
    print(f"Rolling back rows with ts >= {cutoff} ({format_ts(cutoff)})")
    
    count = store.delete_recent(symbol, timeframe, cutoff)
    print(f"Deleted {count} rows.")
    return count


def rollback_recent_limit(
    store: MarketDataStore, symbol: str, timeframe: str, limit: int
) -> int:
    if limit <= 0:
        raise ValueError("limit must be positive")
    
    print(f"Deleting the {limit} most recent rows...")
    count = store.delete_tail(symbol, timeframe, limit)
    print(f"Deleted {count} rows.")
    return count


if __name__ == "__main__":
    main()
