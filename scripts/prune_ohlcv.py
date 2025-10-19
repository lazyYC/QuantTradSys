import argparse
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

MILLIS_PER_DAY = 86_400_000


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prune recent OHLCV data from the SQLite store."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("storage/market_data.db"),
        help="SQLite database path",
    )
    parser.add_argument("--symbol", required=True, help="Trading symbol, e.g. BTC/USDT")
    parser.add_argument("--timeframe", required=True, help="Timeframe, e.g. 5m")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--days", type=float, help="Remove the most recent N days of candles"
    )
    group.add_argument("--limit", type=int, help="Remove the most recent N candles")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show how many rows would be deleted without modifying the DB",
    )
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    try:
        if args.days is not None:
            prune_recent_days(
                conn, args.symbol, args.timeframe, args.days, args.dry_run
            )
        else:
            prune_recent_limit(
                conn, args.symbol, args.timeframe, args.limit, args.dry_run
            )
    finally:
        conn.close()


def _format_ts(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()


def prune_recent_days(
    conn: sqlite3.Connection, symbol: str, timeframe: str, days: float, dry_run: bool
) -> int:
    if days <= 0:
        raise ValueError("days must be positive")
    cursor = conn.execute(
        "SELECT MIN(ts), MAX(ts) FROM ohlcv WHERE symbol = ? AND timeframe = ?",
        (symbol, timeframe),
    )
    row = cursor.fetchone()
    if not row or row[1] is None:
        print("No data found for the specified symbol/timeframe.")
        return 0
    min_ts, max_ts = row
    cutoff = int(max_ts - days * MILLIS_PER_DAY)
    if cutoff <= min_ts:
        cutoff = min_ts
    print(f"Deleting rows with ts >= {cutoff} ({_format_ts(cutoff)})")
    if dry_run:
        cursor = conn.execute(
            "SELECT COUNT(*) FROM ohlcv WHERE symbol = ? AND timeframe = ? AND ts >= ?",
            (symbol, timeframe, cutoff),
        )
        count = cursor.fetchone()[0]
        print(f"[dry-run] Rows that would be deleted: {count}")
        return 0
    cursor = conn.execute(
        "DELETE FROM ohlcv WHERE symbol = ? AND timeframe = ? AND ts >= ?",
        (symbol, timeframe, cutoff),
    )
    conn.commit()
    print(f"Deleted {cursor.rowcount} rows.")
    return cursor.rowcount


def prune_recent_limit(
    conn: sqlite3.Connection, symbol: str, timeframe: str, limit: int, dry_run: bool
) -> int:
    if limit <= 0:
        raise ValueError("limit must be positive")
    cursor = conn.execute(
        "SELECT COUNT(*) FROM ohlcv WHERE symbol = ? AND timeframe = ?",
        (symbol, timeframe),
    )
    total = cursor.fetchone()[0]
    if total == 0:
        print("No data found for the specified symbol/timeframe.")
        return 0
    if limit >= total:
        cutoff = -1
    else:
        cursor = conn.execute(
            "SELECT ts FROM ohlcv WHERE symbol = ? AND timeframe = ? ORDER BY ts DESC LIMIT 1 OFFSET ?",
            (symbol, timeframe, limit - 1),
        )
        row = cursor.fetchone()
        if row is None:
            cutoff = -1
        else:
            cutoff = int(row[0])
    condition = "ts >= ?" if cutoff >= 0 else "1=1"
    params = (symbol, timeframe, cutoff) if cutoff >= 0 else (symbol, timeframe)
    if cutoff >= 0:
        print(f"Deleting rows with ts >= {cutoff} ({_format_ts(cutoff)})")
    else:
        print("Deleting all rows for the specified symbol/timeframe.")
    query = (
        f"SELECT COUNT(*) FROM ohlcv WHERE symbol = ? AND timeframe = ? AND {condition}"
    )
    cursor = conn.execute(query, params)
    count = cursor.fetchone()[0]
    if dry_run:
        print(f"[dry-run] Rows that would be deleted: {count}")
        return 0
    delete_query = (
        f"DELETE FROM ohlcv WHERE symbol = ? AND timeframe = ? AND {condition}"
    )
    cursor = conn.execute(delete_query, params)
    conn.commit()
    print(f"Deleted {cursor.rowcount} rows.")
    return cursor.rowcount


if __name__ == "__main__":
    main()
