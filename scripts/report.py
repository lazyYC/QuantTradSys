"""
Generic reporting script for trading strategies.
Generates HTML reports with performance metrics, equity curves, and trade analysis.
"""

import argparse
import logging
import sys
from pathlib import Path

# 1. Setup (Path, Logging, Config)
# Ensure src is in path for standalone execution
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config.paths import DEFAULT_STATE_DB, DEFAULT_MARKET_DB
from utils.logging import setup_logging
from reporting.engine import ReportEngine

LOGGER = logging.getLogger(__name__)


def main() -> None:
    args = parse_args()
    setup_logging()
    
    try:
        engine = ReportEngine.from_args(args)
        engine.run()
    except Exception as e:
        LOGGER.exception("Report generation failed")
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Strategy Report")
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--study", help="Study name (experiment ID)")
    parser.add_argument("--store-path", type=Path, default=DEFAULT_STATE_DB)
    parser.add_argument("--ohlcv-db", default=DEFAULT_MARKET_DB)
    parser.add_argument("--output", default="reports/report.html")
    parser.add_argument("--title", help="Report Title")
    parser.add_argument("--rerun", action="store_true", help="Force rerun backtest")
    parser.add_argument("--start", help="Start Date (ISO)")
    parser.add_argument("--end", help="End Date (ISO)")
    return parser.parse_args()


if __name__ == "__main__":
    main()

