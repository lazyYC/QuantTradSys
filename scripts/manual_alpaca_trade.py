#!/usr/bin/env python
"""Manual Alpaca trade trigger for testing."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from notifier.dispatcher import (  # noqa: E402
    AlpacaBrokerAdapter,
    execute_trading,
    get_alpaca_client,
    get_latest_price,
)


def main() -> None:
    args = parse_args()

    if args.order_ratio is not None:
        os.environ["ALPACA_ORDER_RATIO"] = str(args.order_ratio)
    if args.price_tolerance is not None:
        os.environ["ALPACA_PRICE_TOLERANCE"] = str(args.price_tolerance)

    client = get_alpaca_client(env_path=args.env)
    if client is None:
        raise SystemExit(
            "Failed to initialize Alpaca client; check credentials or .env path."
        )

    latest_price = get_latest_price(client, args.symbol)
    if latest_price is None:
        raise SystemExit("Unable to fetch latest price; aborting.")

    context = {
        "symbol": args.symbol,
        "price": latest_price,
        "meta": {"source": "manual_test"},
    }
    print("Prepared context:", json.dumps(context, indent=2, default=str))

    if args.dry_run:
        print("[DRY RUN] Order not submitted.")
        return

    broker = AlpacaBrokerAdapter(client)
    success, message = execute_trading(broker, args.action, context)
    print(message or "No execution message.")
    if success:
        print("Order submitted.")
    else:
        print("Order failed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manual Alpaca trade trigger for testing."
    )
    parser.add_argument(
        "--action",
        required=True,
        choices=["ENTER_LONG", "EXIT_LONG", "ENTER_SHORT", "EXIT_SHORT"],
    )
    parser.add_argument("--symbol", default="BTC/USD", help="Trading symbol")
    parser.add_argument(
        "--order-ratio",
        type=float,
        help="Override ALPACA_ORDER_RATIO (fraction of buying power, e.g. 0.95)",
    )
    parser.add_argument(
        "--price-tolerance",
        type=float,
        help="Override ALPACA_PRICE_TOLERANCE (fractional deviation allowed, e.g. 0.005)",
    )
    parser.add_argument(
        "--env",
        type=Path,
        default=SRC_DIR / "config" / ".env",
        help="Path to .env file containing Alpaca credentials",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview order without submitting"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
