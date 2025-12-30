"""
Realtime engine for strategies driven by Binance WebSocket closed klines.
Supports running a specific study of a strategy.
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import sys

# 1. Setup (Path, Logging, Config)
# 1. Setup (Path, Logging, Config)
# Ensure src is in path for standalone execution
# Path hack removed - dependent on pip install -e .


from config.paths import DEFAULT_LOG_DIR
from config.env import load_env
from utils.logging import setup_logging

# Load env immediately
load_env()

from engine.realtime import RealtimeEngine
from persistence.param_store import load_strategy_params
from utils.pid_lock import PIDLock, AlreadyRunningError

LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Realtime strategy engine (websocket driven)"
    )
    parser.add_argument(
        "--strategy", type=str, required=True, help="Strategy algorithm name (e.g. star_xgb)"
    )
    parser.add_argument(
        "--study", type=str, required=True, help="Study name (experiment ID)"
    )
    parser.add_argument(
        "--lookback-days", type=int, default=30, help="History days to load for initialization"
    )
    parser.add_argument("--params-db", type=Path, default=Path("postgres_params"))
    parser.add_argument("--state-db", type=Path, default=Path("postgres_state"))
    parser.add_argument("--ohlcv-db", type=Path, default=Path("postgres_market"))
    parser.add_argument("--exchange", type=str, default="binanceusdm")
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_DIR / "scheduler.log")
    args = parser.parse_args()

    setup_logging(log_path=args.log_path)

    # 1. Load Strategy Params to get Symbol/Timeframe
    record = load_strategy_params(
        args.params_db,
        strategy=args.strategy,
        study=args.study,
        # symbol/timeframe inferred
    )
    
    if record is None:
        LOGGER.error(f"Parameters not found for strategy={args.strategy} study={args.study}")
        sys.exit(1)
        
    symbol = record.symbol
    timeframe = record.timeframe
    
    LOGGER.info(f"Loaded configuration for {args.strategy}/{args.study}: {symbol} {timeframe}")

    # 2. Acquire PID Lock
    lock_file = args.log_path.parent.parent / "locks" / f"{args.strategy}_{args.study}.lock"
    pid_lock = PIDLock(lock_file)
    
    try:
        pid_lock.acquire()
    except AlreadyRunningError as e:
        LOGGER.error(str(e))
        sys.exit(1)

    try:
        engine = RealtimeEngine(
            symbol=symbol,
            timeframe=timeframe,
            strategy=args.strategy,
            study=args.study,
            lookback_days=args.lookback_days,
            params_store_path=args.params_db,
            state_store_path=args.state_db,
            market_db_path=args.ohlcv_db,
            exchange=args.exchange,
            strategy_record=record,
        )
        engine.start()
    finally:
        pid_lock.release()

if __name__ == '__main__':
    main()
