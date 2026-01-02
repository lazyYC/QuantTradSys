"""
Unified training script for QuantTradSys strategies.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# 1. Setup (Path, Logging, Config)

# from config.paths import DEFAULT_STATE_DB # Removed
from training.engine import TrainingEngine
from database.connection import get_database_url

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
LOGGER = logging.getLogger(__name__)


def main() -> None:
    args = _parse_args()
    try:
        engine = TrainingEngine.from_args(args)
        engine.run()
    except Exception as e:
        LOGGER.exception("Training failed")
        sys.exit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and optimize trading strategies.")
    
    parser.add_argument("--strategy", type=str, required=True, help="Strategy to train (must match folder name in src/strategies/)")
    parser.add_argument("--symbol", type=str, default="BTC/USDT:USDT")
    parser.add_argument("--timeframe", type=str, default="5m")
    parser.add_argument("--lookback-days", type=int, default=360)
    parser.add_argument("--exchange", type=str, default="binanceusdm")
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--stop-loss-pct", type=float, default=0.005)
    parser.add_argument("--valid-days", type=int, default=30)
    
    # Target Definition (Fixed)
    parser.add_argument("--future-window", type=int, default=5, help="Fixed prediction horizon (bars)")
    parser.add_argument("--future-return-threshold", type=float, default=0, help="Fixed return threshold")
    
    # Optimization Config
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--n-seeds", type=int, default=5, help="Number of seeds per trial for stability")
    parser.add_argument("--study-name", type=str, default="strategy_optimization")
    
    try:
        default_storage = get_database_url()
    except Exception:
        default_storage = "sqlite:///storage/optuna_studies.db"
        
    parser.add_argument("--storage", type=str, default=default_storage)
    parser.add_argument("--test-days", type=int, default=30, help="Days to use for validation/testing")
    parser.add_argument("--use-gpu", action="store_true")
    
    # Storage Paths
    # parser.add_argument("--store-path", type=Path, default=DEFAULT_STATE_DB, help="Path to SQLite DB for params and trades") # Deprecated
    parser.add_argument("--model-dir", type=Path, default=None, help="Directory to save models (default: storage/models/{strategy})")
    parser.add_argument("--dry-run", action="store_true", help="Run without saving to main DB (uses temp DB)")
    
    return parser.parse_args()


if __name__ == "__main__":
    main()
