#!/usr/bin/env python
"""Simple staging order test against Alpaca crypto endpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from brokers.alpaca import AlpacaPaperTradingClient, AlpacaAPIError
from config.env import DEFAULT_ENV_PATH, load_env
from notifier.dispatcher import format_alpaca_symbol, get_alpaca_client


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit or close an Alpaca paper order for manual verification."
    )
    parser.add_argument("--symbol", type=str, default="BTC/USD")
    parser.add_argument(
        "--side",
        choices=("buy", "sell"),
        help="Side for new market order (ignored when --close is used)",
    )
    parser.add_argument(
        "--notional",
        type=float,
        default=50.0,
        help="Notional USD size for market order (ignored if --qty is provided).",
    )
    parser.add_argument(
        "--qty",
        type=float,
        help="Exact quantity to trade instead of notional.",
    )
    parser.add_argument(
        "--close",
        action="store_true",
        help="Close existing position instead of sending a new order.",
    )
    parser.add_argument(
        "--env-path", type=Path, default=DEFAULT_ENV_PATH, help="Path to .env file."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved request without hitting the API.",
    )
    args = parser.parse_args()

    if not args.close and not args.side:
        parser.error("Either specify --side for a new order or use --close.")

    load_env(args.env_path)
    client = _require_alpaca_client()

    order_symbol = format_alpaca_symbol(args.symbol)
    request_summary: Dict[str, Any] = {
        "symbol": args.symbol,
        "order_symbol": order_symbol,
        "close": args.close,
    }

    if args.close:
        if args.dry_run:
            print("Dry-run close request:", json.dumps(request_summary, indent=2))
            return
        _close_position(client, order_symbol)
        return

    payload: Dict[str, Any] = {
        "symbol": order_symbol,
        "side": args.side,
        "order_type": "market",
        "time_in_force": "gtc",
    }
    if args.qty is not None:
        payload["qty"] = args.qty
    else:
        payload["notional"] = args.notional
    request_summary.update({"side": args.side, "payload": payload})

    if args.dry_run:
        print("Dry-run order payload:", json.dumps(request_summary, indent=2))
        return

    try:
        response = client.submit_order(
            symbol=payload["symbol"],
            side=payload["side"],
            qty=payload.get("qty"),
            notional=payload.get("notional"),
            order_type=payload["order_type"],
            time_in_force=payload["time_in_force"],
        )
    except AlpacaAPIError as exc:
        print("Alpaca API error:", exc)
        if exc.payload:
            print(json.dumps(exc.payload, indent=2))
        return

    print("Order accepted:")
    print(json.dumps(response, indent=2))


def _close_position(client: AlpacaPaperTradingClient, symbol: str) -> None:
    try:
        response = client.close_position(symbol, side=None)
    except AlpacaAPIError as exc:
        print("Failed to close position:", exc)
        if exc.payload:
            print(json.dumps(exc.payload, indent=2))
        return

    print(f"Position close request sent for {symbol}:")
    print(json.dumps(response, indent=2))


def _require_alpaca_client() -> AlpacaPaperTradingClient:
    client = get_alpaca_client()
    if client is None:
        raise RuntimeError("Alpaca client not configured; check API key/secret.")
    return client


if __name__ == "__main__":
    main()
