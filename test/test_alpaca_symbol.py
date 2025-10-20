#!/usr/bin/env python
"""Quick script to verify Alpaca credentials and symbol mapping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from brokers.alpaca import AlpacaPaperTradingClient
from config.env import DEFAULT_ENV_PATH, load_env
from notifier.dispatcher import format_alpaca_symbol, get_alpaca_client


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify Alpaca staging credentials and symbol availability"
    )
    parser.add_argument("--symbol", type=str, default="BTC/USD")
    parser.add_argument(
        "--env-path", type=Path, default=DEFAULT_ENV_PATH, help="Path to .env file"
    )
    args = parser.parse_args()

    load_env(args.env_path)
    client = _require_alpaca_client()

    account = client.account_overview()
    order_symbol = format_alpaca_symbol(args.symbol)
    order_symbol = "BTC/USD"
    asset = client.get_asset(order_symbol)

    print("Alpaca account overview:")
    print(json.dumps(account, indent=2))
    print("\nResolved Alpaca symbol:", order_symbol)
    print("Asset metadata:")
    print(json.dumps(asset, indent=2))


def _require_alpaca_client() -> AlpacaPaperTradingClient:
    client = get_alpaca_client()
    if client is None:
        raise RuntimeError("Alpaca client not configured; check API key/secret")
    return client


if __name__ == "__main__":
    main()
