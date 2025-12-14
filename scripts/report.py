"""
Generic reporting script for trading strategies.
Generates HTML reports with performance metrics, equity curves, and trade analysis.
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Mapping

import pandas as pd
import sqlite3

# 1. Setup (Path, Logging, Config)
import _setup
from _setup import DEFAULT_STATE_DB, DEFAULT_MARKET_DB

FAVICON_PATH = _setup.SRC_DIR / "config" / "favicon.png"

from persistence.param_store import load_strategy_params
from persistence.trade_store import load_metrics, load_trades
from strategies.base import BaseStrategy
from strategies.loader import load_strategy_class
from utils.symbols import canonicalize_symbol
from data_pipeline.reader import read_ohlcv

# Import reporting utilities
from reporting.tables import (
    create_metrics_table,
    create_params_table,
    create_top_trades_table,
    create_trade_distribution_table,
)
from reporting.plotting import build_candlestick_figure, build_trade_overview_figure

LOGGER = logging.getLogger(__name__)


def main() -> None:
    args = parse_args()
    _setup.setup_logging()
    
    # 1. Load Strategy Params & Metadata
    # We load this FIRST to get symbol/timeframe
    study_name = args.study if args.study else "strategy_optimization"
    
    params_record = load_strategy_params(
        args.store_path,
        strategy=args.strategy,
        study=study_name,
        # symbol/timeframe inferred from study
    )
    
    if not params_record:
        LOGGER.error(f"No parameters found for strategy='{args.strategy}' study='{study_name}'. Please train first.")
        sys.exit(1)
        
    symbol = params_record.symbol
    timeframe = params_record.timeframe
    params = params_record.params
    model_path = params_record.model_path
    
    LOGGER.info(f"Loaded study '{study_name}': {symbol} {timeframe}")
    
    # 2. Load Strategy Class
    try:
        strategy_class = load_strategy_class(args.strategy)
    except ValueError as e:
        LOGGER.error(e)
        sys.exit(1)
        
    strategy = strategy_class()

    # 3. Parse Time Boundaries
    start_ts, end_ts = _parse_time_boundaries(args.start, args.end)
    if start_ts and end_ts and start_ts > end_ts:
        LOGGER.error("Start time must be before end time.")
        sys.exit(1)

    # 4. Load Data (OHLCV)
    candles = read_ohlcv(
        Path(args.ohlcv_db),
        symbol,
        timeframe,
        start_ts=start_ts,
        end_ts=end_ts,
    )
    if candles.empty:
        LOGGER.error(f"No OHLCV data found for {symbol} {timeframe}.")
        sys.exit(1)

    # 5. Load Trades & Metrics (Always load 'all' datasets and aggregate)
    trades_df = pd.DataFrame()
    metrics_df = pd.DataFrame()
    sections: List[Mapping[str, object]] = []
    
    force_rerun = args.rerun

    metric_frames: List[pd.DataFrame] = []
    trade_frames: List[pd.DataFrame] = []
    
    # If not forcing rerun, try to load from DB
    if not force_rerun:
        for ds in ("train", "valid", "test"):
            metrics_ds = _load_metrics_from_db(
                args.store_path, args.strategy, study_name, symbol, timeframe,
                dataset_label=ds, start_ts=start_ts, end_ts=end_ts
            )
            if not metrics_ds.empty:
                metrics_local = metrics_ds.copy()
                metrics_local["dataset"] = ds
                metric_frames.append(metrics_local.head(1))
            
            trades_ds = _load_trades_from_db(
                args.store_path, args.strategy, study_name, symbol, timeframe,
                dataset_label=ds, start_ts=start_ts, end_ts=end_ts
            )
            if not trades_ds.empty:
                trades_local = trades_ds.copy()
                trades_local["dataset"] = ds
                trade_frames.append(trades_local)
    
    metrics_df = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    trades_df = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    
    if not trades_df.empty and "entry_time" in trades_df.columns:
        trades_df = trades_df.sort_values("entry_time").reset_index(drop=True)

    # 6. Rerun Backtest if needed
    need_rerun = force_rerun or trades_df.empty or metrics_df.empty
    
    if need_rerun:
        LOGGER.info("Running backtest..." if force_rerun else "No saved data found, running backtest...")
            
        # Run backtest via Strategy Interface
        result = strategy.backtest(candles, params, model_path=model_path)
        
        # Normalize result
        trades_df = result.trades
        metrics_df = pd.DataFrame([result.metrics])
        
        if not trades_df.empty:
            trades_df["dataset"] = "backtest" 
            
    # 7. Build Equity Curve
    equity_df = _build_equity_from_trades(trades_df)
    equity_df = _filter_by_time(equity_df, "timestamp", start_ts, end_ts)

    # 8. Build Sections for Chart (if applicable)
    if not metrics_df.empty:
        if {"period_start", "period_end"}.issubset(metrics_df.columns):
            for _, row in metrics_df.iterrows():
                start_val = row.get("period_start")
                end_val = row.get("period_end")
                dataset_label = row.get("dataset", "Section")
                
                section_start = pd.to_datetime(start_val, utc=True, errors="coerce") if start_val else None
                section_end = pd.to_datetime(end_val, utc=True, errors="coerce") if end_val else None
                
                if pd.notna(section_start) and pd.notna(section_end) and section_end > section_start:
                    sections.append({
                        "label": str(dataset_label).capitalize(),
                        "start": section_start,
                        "end": section_end,
                    })

    # 9. Generate Report
    figures = _collect_figures(
        candles, 
        trades_df, 
        equity_df, 
        metrics_df, 
        params, 
        sections=sections if sections else None
    )
    
    output_path = Path(args.output)
    title = args.title or f"Report: {args.strategy} ({study_name})"
    _write_html(figures, output_path, title=title)
    LOGGER.info(f"Report saved to {output_path}")


def _parse_time_boundaries(start: str | None, end: str | None) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    start_ts = pd.to_datetime(start, utc=True, errors="coerce") if start else None
    end_ts = pd.to_datetime(end, utc=True, errors="coerce") if end else None
    return start_ts, end_ts


def _filter_by_time(df: pd.DataFrame, column: str, start_ts: pd.Timestamp | None, end_ts: pd.Timestamp | None) -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return df
    frame = df.copy()
    frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    frame = frame.dropna(subset=[column])
    if start_ts is not None:
        frame = frame[frame[column] >= start_ts]
    if end_ts is not None:
        frame = frame[frame[column] <= end_ts]
    return frame.reset_index(drop=True)





def _load_trades_from_db(
    store_path: Path,
    strategy: str,
    study: str,
    symbol: str,
    timeframe: str,
    *,
    dataset_label: str,
    start_ts: pd.Timestamp | None = None,
    end_ts: pd.Timestamp | None = None,
) -> pd.DataFrame:
    df = load_trades(
        store_path,
        strategy=strategy,
        study=study,
        dataset=dataset_label,
        symbol=symbol,
        timeframe=timeframe,
    )
    if not df.empty:
        trades = df.copy()
        trades = _filter_by_time(trades, "entry_time", start_ts, end_ts)
        trades = _filter_by_time(trades, "exit_time", start_ts, end_ts)
        return trades
    return pd.DataFrame()


def _load_metrics_from_db(
    store_path: Path,
    strategy: str,
    study: str,
    symbol: str,
    timeframe: str,
    *,
    dataset_label: str,
    start_ts: pd.Timestamp | None = None,
    end_ts: pd.Timestamp | None = None,
) -> pd.DataFrame:
    df = load_metrics(
        store_path,
        strategy=strategy,
        study=study,
        dataset=dataset_label,
        symbol=symbol,
        timeframe=timeframe,
    )
    if not df.empty:
        metrics = df.copy()
        metrics = _filter_by_time(metrics, "created_at", start_ts, end_ts)
        if not metrics.empty:
            return metrics.sort_values("created_at", ascending=False).head(1).reset_index(drop=True)
    return pd.DataFrame()


def _build_equity_from_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or "return" not in trades.columns:
        return pd.DataFrame(columns=["timestamp", "equity"])
    
    closed = trades.dropna(subset=["exit_time", "return"]).copy()
    if closed.empty:
        return pd.DataFrame(columns=["timestamp", "equity"])
        
    closed = closed.sort_values("exit_time")
    returns = pd.to_numeric(closed["return"], errors="coerce").fillna(0.0)
    equity_values = (1 + returns).cumprod()
    
    return pd.DataFrame({"timestamp": closed["exit_time"], "equity": equity_values})


def _collect_figures(
    candles: pd.DataFrame,
    trades: pd.DataFrame,
    equity_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    params: Mapping[str, object],
    *,
    sections: Optional[List[Mapping[str, object]]] = None,
) -> List[Tuple[str, object]]:
    figures: List[Tuple[str, object]] = []
    
    # 1. Overview Chart
    if not candles.empty:
        if not trades.empty or not equity_df.empty:
            overview = build_trade_overview_figure(
                candles, trades, equity=equity_df, show_markers=True
            )
        else:
            overview = build_candlestick_figure(candles, title="Price Overview")
            
        # Add sections (colored backgrounds)
        if sections:
            color_cycle = [
                "rgba(63, 81, 181, 0.12)",
                "rgba(244, 67, 54, 0.12)",
                "rgba(255, 193, 7, 0.12)",
            ]
            sorted_sections = [s for s in sections if s.get("start") and s.get("end")]
            sorted_sections.sort(key=lambda item: item["start"])
            
            for idx, section in enumerate(sorted_sections):
                start = section["start"]
                end = section["end"]
                label = section.get("label")
                color = color_cycle[idx % len(color_cycle)]
                
                overview.add_vrect(
                    x0=start, x1=end,
                    fillcolor=color, opacity=0.35, line_width=0, layer="below"
                )
                mid = start + (end - start) / 2
                overview.add_annotation(
                    x=mid, yref="paper", y=1.05,
                    text=str(label) if label else f"Section {idx+1}",
                    showarrow=False, font=dict(size=11), align="center"
                )
        figures.append(("Price & Trades", overview))
        
    # 2. Params Tables
    # Try to split params into groups if possible, otherwise just one table
    if params:
        # Check for nested params like 'indicator' and 'model' (common in StarXGB)
        if "indicator" in params and isinstance(params["indicator"], dict):
            fig = create_params_table(params["indicator"], title="Indicator Params")
            if fig: figures.append(("Indicator Params", fig))
            
        if "model" in params and isinstance(params["model"], dict):
            fig = create_params_table(params["model"], title="Model Params")
            if fig: figures.append(("Model Params", fig))
            
        # Also show flat params if no nested structure found or as fallback
        if "indicator" not in params and "model" not in params:
             fig = create_params_table(params, title="Strategy Parameters")
             if fig: figures.append(("Parameters", fig))

    # 3. Metrics Table
    if not metrics_df.empty:
        fig = create_metrics_table(metrics_df, title="Performance Metrics")
        if fig: figures.append(("Metrics", fig))
        
    # 4. Trade Distribution
    if not trades.empty:
        fig = create_trade_distribution_table(trades, title="Trade Distribution")
        if fig: figures.append(("Trade Distribution", fig))
        
    # 5. Top Trades
    if not trades.empty:
        fig = create_top_trades_table(trades, top_n=50, title="Top Trades")
        if fig: figures.append(("Trade Log", fig))
        
    return figures


def _write_html(figures: List[Tuple[str, object]], output: Path, title: str) -> None:
    favicon_href: Optional[str] = None
    if FAVICON_PATH.exists():
        try:
            rel_path = os.path.relpath(FAVICON_PATH, output.parent)
            favicon_href = rel_path.replace("\\", "/")
        except ValueError:
            favicon_href = FAVICON_PATH.resolve().as_posix()

    parts = [
        "<html><head>",
        "<meta charset='utf-8'>",
        f"<title>{title}</title>",
    ]
    if favicon_href:
        parts.append(f"<link rel='icon' href='{favicon_href}' type='image/png'>")

    parts.extend([
        (
            "<style>body{font-family:Segoe UI,system-ui,sans-serif;margin:24px;}"
            "h1{margin-bottom:16px;}h2{margin:0 0 12px;}"
            "section.report-block{margin-top:56px;}section.report-block:first-of-type{margin-top:28px;}"
            "section.report-block .plotly-graph-div{max-width:100%;}</style>"
        ),
        "</head><body>",
        f"<h1>{title}</h1>",
    ])

    include_js = True
    for heading, fig in figures:
        parts.append("<section class='report-block'>")
        parts.append(f"<h2>{heading}</h2>")
        parts.append(
            fig.to_html(
                full_html=False, include_plotlyjs="cdn" if include_js else False
            )
        )
        parts.append("</section>")
        include_js = False
    parts.append("</body></html>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(parts), encoding="utf-8")


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

