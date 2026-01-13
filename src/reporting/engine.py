
from __future__ import annotations

import argparse
import logging
import os
import sys
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, List, Optional, Tuple, Any

import pandas as pd

# from config.paths import DEFAULT_STATE_DB, DEFAULT_MARKET_DB # Removed
from persistence.param_store import load_strategy_params
from strategies.base import BaseStrategy
from strategies.loader import load_strategy_class
from persistence.market_store import MarketDataStore
from utils.data_utils import filter_by_time
from utils.trading import build_equity_from_trades
from reporting.tables import (
    create_metrics_table,
    create_params_table,
    create_top_trades_table,
    create_trade_distribution_table,
)
from reporting.plotting import build_candlestick_figure, build_trade_overview_figure, build_scatter_figure

LOGGER = logging.getLogger(__name__)

@dataclass
class ReportContext:
    # Config
    strategy_name: str
    study_name: str

    output_path: Path
    title: Optional[str] = None
    # rerun: bool = True  # Removed, always True now implicitly
    
    # Time Boundaries
    start_ts: Optional[pd.Timestamp] = None
    end_ts: Optional[pd.Timestamp] = None
    
    # Override
    stop_loss_arg: Optional[str] = None

    # Inferred/Loaded Metadata
    symbol: str = ""
    timeframe: str = ""
    params: dict = field(default_factory=dict)
    model_path: Optional[str] = None
    
    # Runtime Data
    candles: pd.DataFrame = field(default_factory=pd.DataFrame)
    trades_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    equity_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    benchmark_equity_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Internal
    sections: List[Mapping[str, object]] = field(default_factory=list)


class ReportEngine:
    def __init__(self, context: ReportContext):
        self.ctx = context
        self.strategy_instance: Optional[BaseStrategy] = None
        self.market_store = MarketDataStore()

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> ReportEngine:
        """Initialize Engine from command line arguments."""
        start_ts = pd.to_datetime(args.start, utc=True, errors="coerce") if args.start else None
        end_ts = pd.to_datetime(args.end, utc=True, errors="coerce") if args.end else None
        
        ctx = ReportContext(
            strategy_name=args.strategy,
            study_name=args.study if args.study else "default_study_name",
            output_path=Path(args.output),
            title=args.title,
            start_ts=start_ts,
            end_ts=end_ts,
            stop_loss_arg=args.stop_loss_pct,
        )
        return cls(ctx)

    def run(self) -> None:
        """Execute the reporting pipeline."""
        LOGGER.info(f"Starting Report Generation for {self.ctx.strategy_name}/{self.ctx.study_name}")
        
        # 1. Prepare Metadata (Params, Symbol, Timeframe)
        self._prepare_metadata()
        
        # 2. Prepare Data (Only Candles, skipping DB trades/metrics)
        self._prepare_data()
        
        # 3. Always Run Backtest
        self._run_backtest()
        
        # 4. Generate Report (Equity, Figures, HTML)
        # 4. Generate Report (Equity, Figures, HTML)
        self._generate_tradingview_report()
        # self._generate_report() # Old Plotly Logic

    def _prepare_metadata(self) -> None:
        """Load strategy params and resolve symbol/timeframe."""
        record = load_strategy_params(
            strategy=self.ctx.strategy_name,
            study=self.ctx.study_name,
        )
        
        if not record:
            LOGGER.error(f"No parameters found for strategy='{self.ctx.strategy_name}' study='{self.ctx.study_name}'. Please train first.")
            sys.exit(1)
            
        self.ctx.symbol = record.symbol
        self.ctx.timeframe = record.timeframe
        self.ctx.params = record.params
        self.ctx.model_path = record.model_path
        
        LOGGER.info(f"Loaded config: {self.ctx.symbol} {self.ctx.timeframe}")
        
        # Validate Time Boundaries
        if self.ctx.start_ts and self.ctx.end_ts and self.ctx.start_ts > self.ctx.end_ts:
            LOGGER.error("Start time must be before end time.")
            sys.exit(1)

        # Resolve Stop Loss Logic
        self._resolve_stop_loss()

    def _resolve_stop_loss(self) -> None:
        """Resolve stop loss percentage based on override or params."""
        override = self.ctx.stop_loss_arg
        final_sl = 0.005 # Fallback default
        
        # 1. Override Check
        if override is not None:
            if str(override).lower() == "false":
                final_sl = 0.0
                LOGGER.info("Stop Loss: DISABLED (Override=False)")
            else:
                try:
                    final_sl = float(override)
                    LOGGER.info(f"Stop Loss: OVERRIDE to {final_sl:.4f}")
                except ValueError:
                    LOGGER.warning(f"Invalid stop-loss arg '{override}', ignoring.")
            
            # Inject into params immediately
            self.ctx.params["stop_loss_pct"] = final_sl
            return

        # 2. Smart Default (Future Return Threshold * 0.5)
        indicator_params = self.ctx.params.get("indicator", {})
        thresh = indicator_params.get("future_return_threshold")
        if not thresh:
            thresh = self.ctx.params.get("future_return_threshold", 0.0)
            
        if thresh:
            try:
                val = float(thresh)
                if val > 0:
                    final_sl = val * 0.5
                    LOGGER.info(f"Stop Loss: Inferred from threshold ({val:.4f} * 0.5 = {final_sl:.4f})")
            except (ValueError, TypeError):
                pass
                
        self.ctx.params["stop_loss_pct"] = final_sl


    def _prepare_data(self) -> None:
        """Load OHLCV data with warmup buffering."""
        # Load Candles
        # We need warmup data for indicators (approx 60 days to be safe for 30-day patterns/windows)
        warmup_days = 5 # Sufficient for 300-500 bars at 5m/15m
        fetch_start = self.ctx.start_ts - pd.Timedelta(days=warmup_days) if self.ctx.start_ts else None
        
        self.ctx.candles = self.market_store.load_candles(
            symbol=self.ctx.symbol,
            timeframe=self.ctx.timeframe,
            start_ts=fetch_start,
            end_ts=self.ctx.end_ts,
        )
        if self.ctx.candles.empty:
            LOGGER.error(f"No OHLCV data found for {self.ctx.symbol} {self.ctx.timeframe}.")
            sys.exit(1)
            
        LOGGER.info(f"Loaded {len(self.ctx.candles)} candles (including warmup from {fetch_start})")

    def _run_backtest(self) -> None:
        """Always rerun backtest."""
        LOGGER.info("Running backtest to generate report data...")

        # Load Strategy
        try:
            strategy_cls = load_strategy_class(self.ctx.strategy_name)
            self.strategy_instance = strategy_cls()
        except ValueError as e:
            LOGGER.error(e)
            sys.exit(1)
            
        # Run Backtest with core_start to trim warmup from results
        result = self.strategy_instance.backtest(
            self.ctx.candles, 
            self.ctx.params, 
            model_path=self.ctx.model_path,
            core_start=self.ctx.start_ts # Add this arg to BaseStrategy.backtest if needed, or pass via kwargs?
        )
        # Note: We need to ensure PlaygroundStrategy.backtest accepts core_start. 
        # Checking adapter.py... it does NOT accept **kwargs explicitly in signature but backtest_star_xgb does.
        # Check adapter.py backtest signature: (self, raw_data, params, model_path).
        # We need to update adapter.py or pass generic kwargs if supported.
        # Let's check adapter.py first. if not, we must update it.
        # Assuming adapter.py needs update too.
        
        # Update Context
        self.ctx.trades_df = result.trades
        if not self.ctx.trades_df.empty:
            self.ctx.trades_df["dataset"] = "backtest"
            
        self.ctx.metrics_df = pd.DataFrame([result.metrics])
        
        # Store Benchmark Equity if available
        if hasattr(result, "benchmark_equity_curve") and result.benchmark_equity_curve is not None:
             self.ctx.benchmark_equity_df = result.benchmark_equity_curve

    # ----------------------------------------------------------------
    # TradingView Logic
    # ----------------------------------------------------------------

    def _generate_tradingview_report(self) -> None:
        """Generate HTML report using TradingView Lightweight Charts."""
        
        # 1. Build Equity Curve
        self.ctx.equity_df = build_equity_from_trades(self.ctx.trades_df, self.ctx.candles)
        self.ctx.equity_df = filter_by_time(self.ctx.equity_df, "timestamp", self.ctx.start_ts, self.ctx.end_ts)
        
        # 2. Prepare Data for JSON
        
        # Candles
        # Candles
        # Trim data for visualization (User doesn't want full 60-day warmup in the chart)
        # Keep a small buffer (e.g. 2 days) for context before the start
        vis_start = self.ctx.start_ts - pd.Timedelta(days=2) if self.ctx.start_ts else None
        
        candles = self.ctx.candles.copy()
        if vis_start:
             candles = candles[candles["timestamp"] >= vis_start]
        
        # LW Charts expects seconds for timestamp
        if "timestamp" in candles.columns:
            candles["time"] = candles["timestamp"].astype('int64') // 10**9
        
        candle_data = []
        volume_data = []
        for _, row in candles.iterrows():
            c = {
                "time": int(row["time"]),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
            }
            candle_data.append(c)
            
            # Volume color based on close > open
            color = "#26a69a" if row["close"] >= row["open"] else "#ef5350"
            v = {
                "time": int(row["time"]),
                "value": float(row["volume"]),
                "color": color
            }
            volume_data.append(v)
            
        # Equity
        equity_data = []
        if not self.ctx.equity_df.empty:
            eq = self.ctx.equity_df.copy()
            eq["time"] = eq["timestamp"].astype('int64') // 10**9
            for _, row in eq.iterrows():
                equity_data.append({
                    "time": int(row["time"]),
                    "value": float(row["equity"])
                })
                
        # Benchmark Equity
        benchmark_data = []
        if not self.ctx.benchmark_equity_df.empty:
             bench_eq = self.ctx.benchmark_equity_df.copy()
             # Ensure timestamp format
             if "timestamp" in bench_eq.columns:
                 bench_eq = filter_by_time(bench_eq, "timestamp", self.ctx.start_ts, self.ctx.end_ts)
                 bench_eq["time"] = pd.to_datetime(bench_eq["timestamp"], utc=True).astype('int64') // 10**9
                 for _, row in bench_eq.iterrows():
                     benchmark_data.append({
                         "time": int(row["time"]),
                         "value": float(row["equity"])
                     })

        # Markers (Trades)
        markers = []
        if not self.ctx.trades_df.empty:
            raw_markers = []
            for _, row in self.ctx.trades_df.iterrows():
                # Entry
                entry_ts = int(pd.to_datetime(row["entry_time"]).timestamp())
                raw_markers.append({
                    "time": entry_ts,
                    "position": "belowBar" if row["side"] == "LONG" else "aboveBar",
                    "color": "#2196F3" if row["side"] == "LONG" else "#E91E63",
                    "shape": "arrowUp" if row["side"] == "LONG" else "arrowDown",
                    "text": f"Entry: {row['entry_price']}",
                    "id": f"entry_{entry_ts}" # Unique key part
                })
                
                # Exit
                exit_ts = int(pd.to_datetime(row["exit_time"]).timestamp())
                raw_markers.append({
                    "time": exit_ts,
                    "position": "aboveBar" if row["side"] == "LONG" else "belowBar", 
                    "color": "#FF9800",
                    "shape": "circle", 
                    "text": f"Exit: {row['exit_price']}",
                    "id": f"exit_{exit_ts}" 
                })
            
            # Group markers to prevent overlap
            marker_map = {}
            for m in raw_markers:
                key = (m["time"], m["position"], m["color"], m["shape"])
                if key not in marker_map:
                    marker_map[key] = []
                marker_map[key].append(m["text"])
            
            for (ts, pos, col, shp), texts in marker_map.items():
                combined_text = " | ".join(texts)
                # If too long, truncate or summarize
                if len(texts) > 2:
                    combined_text = f"{len(texts)} trades | " + " | ".join(texts[:2]) + "..."
                    
                markers.append({
                    "time": ts,
                    "position": pos,
                    "color": col,
                    "shape": shp,
                    "text": combined_text
                })
        
        # Sort markers by time just in case
        markers.sort(key=lambda x: x["time"])
        
        # 4. Prepare Tables (Metrics & Distribution)
        # We render them as simple HTML tables to match the dark theme via CSS
        metrics_html = ""
        if not self.ctx.metrics_df.empty:
            # Transpose for better view if it's a single row
            m_df = self.ctx.metrics_df.copy()
            if len(m_df) == 1:
                m_df = m_df.T.reset_index()
                m_df.columns = ["Metric", "Value"]
            metrics_html = m_df.to_html(classes="data-table", index=False, border=0, float_format=lambda x: f"{x:.4f}")
            
        dist_html = ""
        analysis_html = ""
        trades_html = ""

        if not self.ctx.trades_df.empty:
            t_df = self.ctx.trades_df.copy()
            
            # --- Distribution ---
            dist_data = {
                "Total Trades": len(t_df),
                "Longs": len(t_df[t_df["side"] == "LONG"]),
                "Shorts": len(t_df[t_df["side"] == "SHORT"]),
                "Win Rate": f"{(len(t_df[t_df['return'] > 0]) / len(t_df) * 100):.1f}%" if len(t_df) > 0 else "0%",
                "Avg Return": f"{t_df['return'].mean():.4f}",
            }
            if "holding_mins" in t_df.columns:
                hold_mins = pd.to_numeric(t_df["holding_mins"], errors='coerce').fillna(0)
                dist_data["Holding Avg (m)"] = f"{hold_mins.mean():.1f}"
                dist_data["Holding Median (m)"] = f"{hold_mins.median():.1f}"
                dist_data["Holding Max (m)"] = f"{hold_mins.max():.1f}"
            
            dist_df = pd.DataFrame([dist_data])
            dist_df = dist_df.T.reset_index()
            dist_df.columns = ["Stat", "Value"]
            dist_html = dist_df.to_html(classes="data-table", index=False, border=0)

            # --- Analysis (Correlation) ---
            if "holding_mins" in t_df.columns:
                t_df["return_numeric"] = pd.to_numeric(t_df["return"], errors='coerce').fillna(0)
                t_df["hold_numeric"] = pd.to_numeric(t_df["holding_mins"], errors='coerce').fillna(0)
                
                corr = t_df["hold_numeric"].corr(t_df["return_numeric"])
                
                # Check safe indexing
                long_mask = t_df["side"] == "LONG"
                short_mask = t_df["side"] == "SHORT"
                
                l_corr = t_df.loc[long_mask, "hold_numeric"].corr(t_df.loc[long_mask, "return_numeric"]) if long_mask.any() else 0.0
                s_corr = t_df.loc[short_mask, "hold_numeric"].corr(t_df.loc[short_mask, "return_numeric"]) if short_mask.any() else 0.0
                
                analysis_data = {
                    "Corr (Time vs Return) All": f"{corr:.4f}",
                    "Corr (Time vs Return) Long": f"{l_corr:.4f}",
                    "Corr (Time vs Return) Short": f"{s_corr:.4f}",
                }
                analysis_df = pd.DataFrame([analysis_data]).T.reset_index()
                analysis_df.columns = ["Metric", "Value"]
                analysis_html = analysis_df.to_html(classes="data-table", index=False, border=0)
                
                # Add Scatter Plot
                scatter_fig = build_scatter_figure(t_df, title="Holding Time vs Return")
                if scatter_fig:
                    scatter_html = scatter_fig.to_html(full_html=False, include_plotlyjs='cdn')
                    analysis_html += f"<div style='margin-top: 20px;'>{scatter_html}</div>"

            # --- Full Trade List ---
            # Sort by time
            t_df = t_df.sort_values("entry_time", ascending=False)
            
            # Create Navigation Links for Times
            # Entry Time Link
            t_df["ts_entry"] = pd.to_datetime(t_df["entry_time"]).astype('int64') // 10**9
            t_df["entry_time_display"] = pd.to_datetime(t_df["entry_time"]).dt.strftime("%Y-%m-%d %H:%M")
            t_df["entry_time"] = t_df.apply(
                lambda row: f"<span class='time-link' onclick='window.scrollToTimestamp({row['ts_entry']})'>{row['entry_time_display']}</span>", 
                axis=1
            )
            
            # Exit Time Link
            if "exit_time" in t_df.columns:
                t_df["ts_exit"] = pd.to_datetime(t_df["exit_time"]).astype('int64') // 10**9
                t_df["exit_time_display"] = pd.to_datetime(t_df["exit_time"]).dt.strftime("%Y-%m-%d %H:%M")
                t_df["exit_time"] = t_df.apply(
                    lambda row: f"<span class='time-link' onclick='window.scrollToTimestamp({row['ts_exit']})'>{row['exit_time_display']}</span>", 
                    axis=1
                )

            # Format columns
            view_cols = ["entry_time", "exit_time", "side", "entry_price", "exit_price", "return", "holding_mins", "exit_reason"]
            # Ensure columns exist
            view_cols = [c for c in view_cols if c in t_df.columns]
            
            view_df = t_df[view_cols].copy()
            # Note: Dates are already formatted as HTML strings above
            
            if "return" in view_df.columns:
                view_df["return"] = view_df["return"].map(lambda x: f"{float(x):.4f}" if pd.notna(x) else "-")
            if "holding_mins" in view_df.columns:
                view_df["holding_mins"] = view_df["holding_mins"].map(lambda x: f"{float(x):.1f}" if pd.notna(x) else "-")
            if "entry_price" in view_df.columns:
                view_df["entry_price"] = view_df["entry_price"].map(lambda x: f"{float(x):.2f}" if pd.notna(x) else "-")
            if "exit_price" in view_df.columns:
                view_df["exit_price"] = view_df["exit_price"].map(lambda x: f"{float(x):.2f}" if pd.notna(x) else "-")

            trades_html = view_df.to_html(classes="data-table", index=False, border=0, escape=False)
        
        # 5. Read Template and Inject
        template_path = Path(__file__).parent / "templates" / "tradingview_report.html"
        if not template_path.exists():
            LOGGER.error(f"Template not found at {template_path}")
            return
            
        html_content = template_path.read_text(encoding="utf-8")
        
        # Calculate Interval Seconds for Zoom
        interval_s = 300 # Default 5m
        tf = self.ctx.timeframe.lower()
        if tf.endswith('m'):
            interval_s = int(tf[:-1]) * 60
        elif tf.endswith('h'):
            interval_s = int(tf[:-1]) * 3600
        elif tf.endswith('d'):
            interval_s = int(tf[:-1]) * 86400
            
        # Replacements
        replacements = {
            "/*TITLE*/": f"Strategy Report: {self.ctx.strategy_name}",
            "/*STRATEGY*/": self.ctx.strategy_name,
            "/*STUDY*/": self.ctx.study_name,
            "/*INJECT_CANDLES*/": json.dumps(candle_data),
            "/*INJECT_VOLUME*/": json.dumps(volume_data),
            "/*INJECT_EQUITY*/": json.dumps(equity_data),
            "/*INJECT_BENCHMARK_EQUITY*/": json.dumps(benchmark_data),
            "/*INJECT_MARKERS*/": json.dumps(markers),
            "/*INJECT_METRICS*/": metrics_html,
            "/*INJECT_DISTRIBUTION*/": dist_html,
            "/*INJECT_ANALYSIS*/": analysis_html,
            "/*INJECT_TRADES*/": trades_html,
            "/*INJECT_INTERVAL*/": str(interval_s)
        }
        
        for key, val in replacements.items():
            html_content = html_content.replace(key, val)
            
        # 6. Write Output
        self.ctx.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.ctx.output_path.write_text(html_content, encoding="utf-8")
        LOGGER.info(f"Report saved to {self.ctx.output_path}")

    def _generate_report(self) -> None:
        """Calculate Equity, Build Figures, Write HTML."""
        # 1. Build Equity Curve
        self.ctx.equity_df = build_equity_from_trades(self.ctx.trades_df, self.ctx.candles)
        self.ctx.equity_df = filter_by_time(self.ctx.equity_df, "timestamp", self.ctx.start_ts, self.ctx.end_ts)

        # 2. Build Sections (Background colors for datasets)
        self._build_sections()

        # 3. Collect Figures
        figures = self._collect_figures()
        
        # 4. Write HTML
        title = self.ctx.title or f"Report: {self.ctx.strategy_name} ({self.ctx.study_name})"
        self._write_html(figures, self.ctx.output_path, title)
        LOGGER.info(f"Report saved to {self.ctx.output_path}")

    # ----------------------------------------------------------------
    # Internal Logic
    # ----------------------------------------------------------------

    def _build_sections(self) -> None:
        if self.ctx.metrics_df.empty:
            return
        if not {"period_start", "period_end"}.issubset(self.ctx.metrics_df.columns):
            return

        for _, row in self.ctx.metrics_df.iterrows():
            start_val = row.get("period_start")
            end_val = row.get("period_end")
            dataset_label = row.get("dataset", "Section")
            
            section_start = pd.to_datetime(start_val, utc=True, errors="coerce") if start_val else None
            section_end = pd.to_datetime(end_val, utc=True, errors="coerce") if end_val else None
            
            if pd.notna(section_start) and pd.notna(section_end) and section_end > section_start:
                self.ctx.sections.append({
                    "label": str(dataset_label).capitalize(),
                    "start": section_start,
                    "end": section_end,
                })

    def _collect_figures(self) -> List[Tuple[str, object]]:
        figures = []
        
        # 1. Overview
        if not self.ctx.candles.empty:
            if not self.ctx.trades_df.empty or not self.ctx.equity_df.empty:
                overview = build_trade_overview_figure(
                    self.ctx.candles, self.ctx.trades_df, equity=self.ctx.equity_df, show_markers=True
                )
            else:
                overview = build_candlestick_figure(self.ctx.candles, title="Price Overview")
                
            # Add sections
            if self.ctx.sections:
                color_cycle = [
                    "rgba(63, 81, 181, 0.12)",
                    "rgba(244, 67, 54, 0.12)",
                    "rgba(255, 193, 7, 0.12)",
                ]
                sorted_sections = [s for s in self.ctx.sections if s.get("start") and s.get("end")]
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

        # 2. Params
        params = self.ctx.params
        if params:
            if "indicator" in params and isinstance(params["indicator"], dict):
                fig = create_params_table(params["indicator"], title="Indicator Params")
                if fig: figures.append(("Indicator Params", fig))
            if "model" in params and isinstance(params["model"], dict):
                fig = create_params_table(params["model"], title="Model Params")
                if fig: figures.append(("Model Params", fig))
            if "indicator" not in params and "model" not in params:
                 fig = create_params_table(params, title="Strategy Parameters")
                 if fig: figures.append(("Parameters", fig))

        # 3. Metrics
        if not self.ctx.metrics_df.empty:
            fig = create_metrics_table(self.ctx.metrics_df, title="Performance Metrics")
            if fig: figures.append(("Metrics", fig))

        # 4. Distribution
        if not self.ctx.trades_df.empty:
            fig = create_trade_distribution_table(self.ctx.trades_df, title="Trade Distribution")
            if fig: figures.append(("Trade Distribution", fig))
            
        # 5. Top Trades
        if not self.ctx.trades_df.empty:
            fig = create_top_trades_table(self.ctx.trades_df, top_n=50, title="Top Trades")
            if fig: figures.append(("Trade Log", fig))
            
        return figures

    def _write_html(self, figures: List[Tuple[str, object]], output: Path, title: str) -> None:
        # Determine Facicon Path relation to output
        favicon_href = None
        try:
             src_dir = Path(__file__).resolve().parent.parent
             favicon_path = src_dir / "config" / "favicon.png"
             if favicon_path.exists():
                 try:
                    rel_path = os.path.relpath(favicon_path, output.parent)
                    favicon_href = rel_path.replace("\\", "/")
                 except ValueError:
                    favicon_href = favicon_path.resolve().as_posix()
        except Exception:
             pass

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
