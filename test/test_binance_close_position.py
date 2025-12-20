"""Binance USD-M 關閉部位測試工具。"""

from __future__ import annotations

import argparse
import logging
import os
from pprint import pformat
from typing import Optional

from brokers.binance import (
    BinanceAPIError,
    BinanceCredentials,
    BinanceUSDMClient,
)
from config.env import DEFAULT_ENV_PATH, load_env
from notifier.dispatcher import (
    BinanceBrokerAdapter,
    _configure_binance_symbol
)
from utils.symbols import canonicalize_symbol

LOGGER = logging.getLogger("binance_close_tester")


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, *, default: Optional[int] = None) -> Optional[int]:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def build_binance_client() -> BinanceUSDMClient:
    api_key = os.getenv("BINANCE_API_KEY") or os.getenv("BINANCE_API_KEY_ID")
    api_secret = os.getenv("BINANCE_API_SECRET") or os.getenv("BINANCE_PRIVATE_KEY")
    if not api_key or not api_secret:
        raise SystemExit("缺少 BINANCE_API_KEY / BINANCE_API_SECRET，無法建立客戶端")

    sandbox = _get_bool("BINANCE_SANDBOX", True)
    recv_window = _get_int("BINANCE_RECV_WINDOW", default=5000) or 5000
    default_symbol = os.getenv("BINANCE_DEFAULT_SYMBOL")

    client = BinanceUSDMClient(
        BinanceCredentials(api_key=api_key, api_secret=api_secret),
        sandbox=sandbox,
        recv_window=recv_window,
        default_symbol=default_symbol,
    )
    _configure_binance_symbol(client)
    return client


def main() -> None:
    parser = argparse.ArgumentParser(
        description="檢查並測試 Binance USD-M close_position 功能"
    )
    parser.add_argument("--env", type=str, default=str(DEFAULT_ENV_PATH))
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Override feed symbol; falls back to BINANCE_SYMBOL or BTC/USDT:USDT",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="實際送出關倉單（預設僅檢視部位資訊）",
    )
    parser.add_argument(
        "--side",
        type=str,
        choices=("buy", "sell"),
        default=None,
        help="可選擇強制使用 buy/sell 關倉，未指定則由系統判定",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    load_env(args.env)
    env_symbol = os.getenv("BINANCE_SYMBOL")
    symbol_input = args.symbol or env_symbol or "BTC/USDT:USDT"
    symbol = canonicalize_symbol(symbol_input)
    client = build_binance_client()
    adapter = BinanceBrokerAdapter(client)
    order_symbol = adapter.format_order_symbol(symbol)

    LOGGER.info("查看部位 | feed_symbol=%s order_symbol=%s", symbol, order_symbol)
    position = client.get_position(order_symbol)
    if not position:
        LOGGER.warning("找不到任何開倉部位")
    else:
        LOGGER.info("目前部位資訊:\n%s", pformat(position))

    if not args.execute:
        LOGGER.info("僅檢視模式 (--execute 未指定)，不送出關倉指令")
        return

    LOGGER.info("送出 close_position 指令 (side=%s)", args.side or "auto")
    try:
        result = client.close_position(order_symbol, side=args.side)
    except BinanceAPIError as exc:
        LOGGER.error("關倉失敗: %s", exc)
        raise SystemExit(1) from exc

    LOGGER.info("Binance API 回應:\n%s", pformat(result))


if __name__ == "__main__":
    main()
