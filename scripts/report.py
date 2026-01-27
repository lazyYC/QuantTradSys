"""
Generic reporting script for trading strategies.
Generates HTML reports with performance metrics, equity curves, and trade analysis.
"""

import argparse
import logging
import sys
from pathlib import Path

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

    parser.add_argument("--output", default="reports/report.html")
    parser.add_argument("--title", help="Report Title")
    parser.add_argument("--start", help="Start Date (ISO)")
    parser.add_argument("--end", help="End Date (ISO)")
    parser.add_argument("--last-days", type=int, help="Run backtest on last N days (aligns with Grid Search)", default=None)
    parser.add_argument("--stop-loss-pct", help="Override Stop Loss % (e.g. 0.005 or 'false')", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()

