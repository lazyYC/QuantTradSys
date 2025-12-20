
from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, List, Optional, Tuple, Any

import pandas as pd

from config.paths import DEFAULT_STATE_DB, DEFAULT_MARKET_DB
from persistence.param_store import load_strategy_params
from persistence.trade_store import load_metrics, load_trades
from strategies.base import BaseStrategy
from strategies.loader import load_strategy_class
from data_pipeline.reader import read_ohlcv
from reporting.tables import (
    create_metrics_table,
    create_params_table,
    create_top_trades_table,
    create_trade_distribution_table,
)
from reporting.plotting import build_candlestick_figure, build_trade_overview_figure

LOGGER = logging.getLogger(__name__)

@dataclass
class ReportContext:
    # Config
    strategy_name: str
    study_name: str
    store_path: Path
    ohlcv_db_path: Path
    output_path: Path
    title: Optional[str] = None
    rerun: bool = False
    
    # Time Boundaries
    start_ts: Optional[pd.Timestamp] = None
    end_ts: Optional[pd.Timestamp] = None

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
    
    # Internal
    sections: List[Mapping[str, object]] = field(default_factory=list)


class ReportEngine:
    def __init__(self, context: ReportContext):
        self.ctx = context
        self.strategy_instance: Optional[BaseStrategy] = None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> ReportEngine:
        """Initialize Engine from command line arguments."""
        start_ts = pd.to_datetime(args.start, utc=True, errors="coerce") if args.start else None
        end_ts = pd.to_datetime(args.end, utc=True, errors="coerce") if args.end else None
        
        ctx = ReportContext(
            strategy_name=args.strategy,
            study_name=args.study if args.study else "default_study_name",
            store_path=Path(args.store_path),
            ohlcv_db_path=Path(args.ohlcv_db),
            output_path=Path(args.output),
            title=args.title,
            rerun=args.rerun,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        return cls(ctx)

    def run(self) -> None:
        """Execute the reporting pipeline."""
        LOGGER.info(f"Starting Report Generation for {self.ctx.strategy_name}/{self.ctx.study_name}")
        
        # 1. Prepare Metadata (Params, Symbol, Timeframe)
        self._prepare_metadata()
        
        # 2. Prepare Data (Candles, Trades, Metrics)
        self._prepare_data()
        
        # 3. Rerun Backtest if needed
        self._run_backtest_if_needed()
        
        # 4. Generate Report (Equity, Figures, HTML)
        self._generate_report()

    def _prepare_metadata(self) -> None:
        """Load strategy params and resolve symbol/timeframe."""
        record = load_strategy_params(
            self.ctx.store_path,
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

    def _prepare_data(self) -> None:
        """Load OHLCV and existing Trades/Metrics from DB."""
        # Load Candles
        self.ctx.candles = read_ohlcv(
            self.ctx.ohlcv_db_path,
            self.ctx.symbol,
            self.ctx.timeframe,
            start_ts=self.ctx.start_ts,
            end_ts=self.ctx.end_ts,
        )
        if self.ctx.candles.empty:
            LOGGER.error(f"No OHLCV data found for {self.ctx.symbol} {self.ctx.timeframe}.")
            sys.exit(1)

        # Skip DB load if rerun is forced
        if self.ctx.rerun:
            return

        # Load Existing Trades & Metrics
        metric_frames = []
        trade_frames = []
        
        for ds in ("train", "valid", "test"):
            # Load Metrics
            m_df = load_metrics(
                self.ctx.store_path,
                strategy=self.ctx.strategy_name,
                study=self.ctx.study_name,
                dataset=ds,
                symbol=self.ctx.symbol,
                timeframe=self.ctx.timeframe,
            )
            if not m_df.empty:
                m_filtered = self._filter_by_time(m_df, "created_at")
                if not m_filtered.empty:
                    m_latest = m_filtered.sort_values("created_at", ascending=False).head(1).copy()
                    m_latest["dataset"] = ds
                    metric_frames.append(m_latest)

            # Load Trades
            t_df = load_trades(
                self.ctx.store_path,
                strategy=self.ctx.strategy_name,
                study=self.ctx.study_name,
                dataset=ds,
                symbol=self.ctx.symbol,
                timeframe=self.ctx.timeframe,
            )
            if not t_df.empty:
                t_filtered = self._filter_by_time(t_df, "entry_time") # Basic filter
                t_filtered = self._filter_by_time(t_filtered, "exit_time") # Ensure exit info valid
                if not t_filtered.empty:
                    t_local = t_filtered.copy()
                    t_local["dataset"] = ds
                    trade_frames.append(t_local)

        if metric_frames:
            self.ctx.metrics_df = pd.concat(metric_frames, ignore_index=True)
        if trade_frames:
            self.ctx.trades_df = pd.concat(trade_frames, ignore_index=True)
            if "entry_time" in self.ctx.trades_df.columns:
                self.ctx.trades_df = self.ctx.trades_df.sort_values("entry_time").reset_index(drop=True)

    def _run_backtest_if_needed(self) -> None:
        """Rerun backtest if data missing or rerun requested."""
        if not self.ctx.rerun and not self.ctx.trades_df.empty and not self.ctx.metrics_df.empty:
            return
            
        LOGGER.info("Running backtest..." if self.ctx.rerun else "No saved data found, running backtest...")

        # Load Strategy
        try:
            strategy_cls = load_strategy_class(self.ctx.strategy_name)
            self.strategy_instance = strategy_cls()
        except ValueError as e:
            LOGGER.error(e)
            sys.exit(1)
            
        # Run Backtest
        result = self.strategy_instance.backtest(
            self.ctx.candles, 
            self.ctx.params, 
            model_path=self.ctx.model_path
        )
        
        # Update Context
        self.ctx.trades_df = result.trades
        if not self.ctx.trades_df.empty:
            self.ctx.trades_df["dataset"] = "backtest"
            
        self.ctx.metrics_df = pd.DataFrame([result.metrics])

    def _generate_report(self) -> None:
        """Calculate Equity, Build Figures, Write HTML."""
        # 1. Build Equity Curve
        self.ctx.equity_df = self._build_equity_from_trades(self.ctx.trades_df, self.ctx.candles)
        self.ctx.equity_df = self._filter_by_time(self.ctx.equity_df, "timestamp")

        # 2. Build Sections (Background colors for datasets)
        self._build_sections()

        # 3. Collect Figures
        figures = self._collect_figures()
        
        # 4. Write HTML
        title = self.ctx.title or f"Report: {self.ctx.strategy_name} ({self.ctx.study_name})"
        self._write_html(figures, self.ctx.output_path, title)
        LOGGER.info(f"Report saved to {self.ctx.output_path}")

    # ----------------------------------------------------------------
    # Internal Logic (Moved from report.py)
    # ----------------------------------------------------------------

    def _filter_by_time(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        if df.empty or column not in df.columns:
            return df
        frame = df.copy()
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
        frame = frame.dropna(subset=[column])
        if self.ctx.start_ts is not None:
            frame = frame[frame[column] >= self.ctx.start_ts]
        if self.ctx.end_ts is not None:
            frame = frame[frame[column] <= self.ctx.end_ts]
        return frame.reset_index(drop=True)

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

    def _build_equity_from_trades(self, trades: pd.DataFrame, candles: pd.DataFrame = None) -> pd.DataFrame:
        # Ported logic from original report.py
        if candles is None or candles.empty:
             # Fallback simple
             if trades.empty or "return" not in trades.columns:
                return pd.DataFrame(columns=["timestamp", "equity"])
             closed = trades.dropna(subset=["exit_time", "return"]).copy()
             if closed.empty:
                return pd.DataFrame(columns=["timestamp", "equity"])
             closed = closed.sort_values("exit_time")
             returns = pd.to_numeric(closed["return"], errors="coerce").fillna(0.0)
             equity_values = (1 + returns).cumprod()
             return pd.DataFrame({"timestamp": closed["exit_time"], "equity": equity_values})

        if trades.empty:
             return pd.DataFrame({"timestamp": candles["timestamp"], "equity": 1.0})

        # Align times
        df = candles[["timestamp", "close"]].copy().sort_values("timestamp").drop_duplicates("timestamp")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
        # df["close"] = pd.to_numeric(df["close"], errors="coerce").ffill() # Assuming already float or handled

        longs = trades[trades["side"] == "LONG"]
        shorts = trades[trades["side"] == "SHORT"]

        def _agg_counts(frame, sign, col):
            if frame.empty: return None
            times = pd.to_datetime(frame[col], utc=True)
            counts = times.value_counts()
            return counts * sign

        l_in = _agg_counts(longs, 1, "entry_time")
        l_out = _agg_counts(longs, -1, "exit_time")
        s_in = _agg_counts(shorts, -1, "entry_time")
        s_out = _agg_counts(shorts, 1, "exit_time")

        all_changes = pd.concat([l_in, l_out, s_in, s_out], axis=0)
        # Handle all None
        if all_changes.empty:
             return pd.DataFrame({"timestamp": candles["timestamp"], "equity": 1.0})
             
        all_changes = all_changes.groupby(level=0).sum()
        
        aligned_changes = all_changes.reindex(df.index, fill_value=0)
        current_pos = aligned_changes.cumsum()
        exposure = current_pos.shift(1).fillna(0)
        
        price_ret = df["close"].pct_change().fillna(0)
        strat_ret = exposure * price_ret
        
        cost_rate = 0.001 
        volume = aligned_changes.abs()
        costs = volume * cost_rate
        total_ret = strat_ret - costs
        
        equity = (1 + total_ret).cumprod()
        return pd.DataFrame({"timestamp": equity.index, "equity": equity.values})

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
        # Assuming run from root, so we know where src is
        # Actually prefer explicit SRC_DIR pass or resolve relative
        # Simplification: use generic or embed?
        # Let's try to resolve like original script
        try:
             # Hacky: Assume engine.py is in src/reporting/
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
