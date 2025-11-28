"""
Generic reporting script for trading strategies.
Generates HTML reports with performance metrics, equity curves, and trade analysis.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Add src to path
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from persistence.param_store import load_strategy_params
from persistence.trade_store import load_metrics, load_trades
from strategies.base import BaseStrategy
from strategies.star_xgb.adapter import StarXGBStrategy
from utils.symbols import canonicalize_symbol

# Import reporting utilities (reusing existing ones for now)
# We might need to refactor these to be more generic if they are star_xgb specific
from reporting.tables import (
    create_metrics_table,
    create_params_table,
    create_top_trades_table,
    create_trade_distribution_table,
)
from reporting.plotting import build_candlestick_figure, build_trade_overview_figure

# Strategy Registry
STRATEGIES: Dict[str, type[BaseStrategy]] = {
    "star_xgb": StarXGBStrategy,
}

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def main() -> None:
    args = parse_args()
    
    # 1. Load Strategy Class
    if args.strategy not in STRATEGIES:
        # Try to infer from args.strategy name (e.g. star_xgb_default -> star_xgb)
        # But for now require exact match or mapping
        # Let's assume the user passes the key in STRATEGIES
        # If args.strategy is "star_xgb_default", we might need to map it.
        # For now, let's just use "star_xgb" as the key.
        if "star_xgb" in args.strategy:
             strategy_class = STRATEGIES["star_xgb"]
        else:
             raise ValueError(f"Unknown strategy: {args.strategy}")
    else:
        strategy_class = STRATEGIES[args.strategy]
        
    strategy = strategy_class()

    # 2. Load Data (OHLCV)
    # Reuse logic from render_star_xgb_report.py or simplify
    # For brevity, I'll assume we load from DB
    candles = _load_candles(args)
    if candles.empty:
        LOGGER.error("No OHLCV data found.")
        sys.exit(1)

    # 3. Load Params & Previous Results
    params_record = load_strategy_params(
        args.store_path,
        strategy=args.strategy,
        symbol=args.symbol,
        timeframe=args.timeframe,
    )
    
    params = params_record.params if params_record else {}
    
    # 4. Determine if Rerun Needed
    trades_df = load_trades(
        args.store_path,
        strategy=args.strategy,
        symbol=args.symbol,
        timeframe=args.timeframe,
    )
    
    metrics_df = load_metrics(
        args.store_path,
        strategy=args.strategy,
        symbol=args.symbol,
        timeframe=args.timeframe,
    )
    
    if args.rerun or trades_df.empty:
        LOGGER.info("Running backtest...")
        if not params:
            LOGGER.error("No parameters found for backtest. Please train first.")
            sys.exit(1)
            
        # Run backtest via Strategy Interface
        # Note: backtest might return different things depending on strategy
        # But we expect a result object with trades, metrics, equity
        result = strategy.backtest(candles, params)
        
        # Normalize result
        # Assuming result has .trades (DataFrame), .metrics (Dict), .equity_curve (DataFrame)
        trades_df = result.trades
        metrics_df = pd.DataFrame([result.metrics])
        equity_df = result.equity_curve
        
        # We don't necessarily save here unless requested, but report generation needs them.
    else:
        LOGGER.info("Loaded trades and metrics from DB.")
        equity_df = _build_equity_from_trades(trades_df)

    # 5. Generate Report
    figures = _collect_figures(candles, trades_df, equity_df, metrics_df, params)
    
    output_path = Path(args.output)
    _write_html(figures, output_path, title=args.title or f"Report: {args.strategy}")
    LOGGER.info(f"Report saved to {output_path}")


def _load_candles(args) -> pd.DataFrame:
    # Simplified loading logic
    import sqlite3
    conn = sqlite3.connect(args.ohlcv_db)
    query = "SELECT ts, open, high, low, close, volume FROM ohlcv WHERE symbol=? AND timeframe=? ORDER BY ts"
    df = pd.read_sql_query(query, conn, params=(args.symbol, args.timeframe))
    conn.close()
    
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.drop(columns=["ts"])
        
        # Filter by date if provided
        if args.start:
            df = df[df["timestamp"] >= pd.to_datetime(args.start, utc=True)]
        if args.end:
            df = df[df["timestamp"] <= pd.to_datetime(args.end, utc=True)]
            
    return df

def _build_equity_from_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or "return" not in trades.columns:
        return pd.DataFrame()
    
    trades = trades.sort_values("exit_time")
    returns = pd.to_numeric(trades["return"], errors="coerce").fillna(0.0)
    equity = (1 + returns).cumprod()
    return pd.DataFrame({"timestamp": trades["exit_time"], "equity": equity})

def _collect_figures(candles, trades, equity, metrics, params):
    figures = []
    
    # Overview
    if not candles.empty:
        fig = build_trade_overview_figure(candles, trades, equity=equity, show_markers=True)
        figures.append(("Price & Trades", fig))
        
    # Metrics
    if not metrics.empty:
        fig = create_metrics_table(metrics)
        figures.append(("Metrics", fig))
        
    # Params
    if params:
        # Flatten params for display if nested
        # For now just dump
        pass 
        
    # Trades
    if not trades.empty:
        fig = create_top_trades_table(trades)
        figures.append(("Top Trades", fig))
        
    return figures

def _write_html(figures, output_path, title):
    html_parts = [
        f"<html><head><title>{title}</title>",
        "<style>body{font-family:sans-serif;margin:20px;}</style>",
        "<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>",
        "</head><body>",
        f"<h1>{title}</h1>"
    ]
    
    for name, fig in figures:
        html_parts.append(f"<h2>{name}</h2>")
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs=False))
        
    html_parts.append("</body></html>")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(html_parts), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Strategy Report")
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--symbol", default="BTC/USDT:USDT")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--store-path", default="storage/strategy_state.db")
    parser.add_argument("--ohlcv-db", default="storage/market_data.db")
    parser.add_argument("--output", default="reports/report.html")
    parser.add_argument("--title", help="Report Title")
    parser.add_argument("--rerun", action="store_true", help="Force rerun backtest")
    parser.add_argument("--start", help="Start Date")
    parser.add_argument("--end", help="End Date")
    return parser.parse_args()

if __name__ == "__main__":
    main()
