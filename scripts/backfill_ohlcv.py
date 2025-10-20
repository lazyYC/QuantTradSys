import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from data_pipeline.ccxt_fetcher import (
    configure_logging,
    ensure_database,
    fetch_yearly_ohlcv,
    prune_older_rows,
    upsert_ohlcv_rows,
)

LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill OHLCV data into SQLite store"
    )
    parser.add_argument("symbol", help="symbol, e.g. BTC/USD")
    parser.add_argument("timeframe", help="timeframe, e.g. 5m")
    parser.add_argument("lookback_days", type=int, help="days to backfill")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("storage/market_data.db"),
        help="SQLite database path",
    )
    parser.add_argument(
        "--exchange", default="binance", help="Exchange id supported by ccxt"
    )
    parser.add_argument(
        "--prune", action="store_true", help="Prune older rows beyond lookback window"
    )
    args = parser.parse_args()

    configure_logging()
    LOGGER.info(
        "Fetching %s %s for %s days", args.symbol, args.timeframe, args.lookback_days
    )
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
    rows = dataframe_to_rows(df)
    inserted = upsert_ohlcv_rows(conn, args.symbol, args.timeframe, rows)
    LOGGER.info("Inserted %s rows into %s", inserted, args.db)

    if args.prune:
        utc_now = datetime.now(timezone.utc)
        keep_from_ms = int(
            (utc_now.timestamp() * 1000) - args.lookback_days * 86400 * 1000
        )
        pruned = prune_older_rows(conn, args.symbol, args.timeframe, keep_from_ms)
        LOGGER.info("Pruned %s old rows", pruned)

    conn.close()


def dataframe_to_rows(df: pd.DataFrame) -> list[tuple]:
    rows: list[tuple] = []
    for _, row in df.iterrows():
        timestamp = pd.Timestamp(row["timestamp"])
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        ts = int(timestamp.timestamp() * 1000)
        iso_ts = timestamp.isoformat().replace("+00:00", "Z")
        rows.append(
            (
                ts,
                iso_ts,
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                float(row["volume"]),
            )
        )
    return rows


if __name__ == "__main__":
    main()
