"""Plotly figure utilities for strategy reports."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLOR_LONG = "#16a34a"
COLOR_SHORT = "#dc2626"
COLOR_EXIT = "#f97316"


def _prepare_candle_frame(candles: pd.DataFrame) -> pd.DataFrame:
    """Normalise candle schema and ensure numeric OHLC values."""

    if candles is None or candles.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])

    frame = candles.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")

    for col in ["open", "high", "low", "close"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    if "open" in frame.columns:
        frame["open"] = frame["open"].ffill()
        if "close" in frame.columns:
            frame["open"] = frame["open"].fillna(frame["close"])
    if "high" in frame.columns and "close" in frame.columns:
        frame["high"] = frame["high"].fillna(frame["close"])
    if "low" in frame.columns and "close" in frame.columns:
        frame["low"] = frame["low"].fillna(frame["close"])

    frame = frame.dropna(subset=["timestamp", "open", "high", "low", "close"])
    return frame.reset_index(drop=True)


def _build_event_strings(
    candles: pd.DataFrame, trades: Optional[pd.DataFrame]
) -> List[str]:
    """Collect entry/exit descriptions keyed by candle timestamp."""

    if trades is None or trades.empty or candles.empty:
        return ["" for _ in range(len(candles))]

    events: Dict[pd.Timestamp, List[str]] = {}

    def _register(ts: pd.Timestamp, message: str) -> None:
        if ts is pd.NaT or pd.isna(ts):
            return
        events.setdefault(ts, []).append(message)

    trade_frame = trades.copy()
    trade_frame["entry_time"] = pd.to_datetime(
        trade_frame.get("entry_time"), utc=True, errors="coerce"
    )
    trade_frame["exit_time"] = pd.to_datetime(
        trade_frame.get("exit_time"), utc=True, errors="coerce"
    )

    for _, row in trade_frame.iterrows():
        side = (row.get("side") or "?").upper()
        entry_price = row.get("entry_price")
        entry_time = row.get("entry_time")
        if pd.notna(entry_time):
            label = f"Entry {side}"
            if pd.notna(entry_price):
                label += f" @ {entry_price:.2f}"
            _register(entry_time, label)

        exit_reason = row.get("exit_reason") or "Exit"
        exit_price = row.get("exit_price")
        exit_time = row.get("exit_time")
        if pd.notna(exit_time):
            label = f"Exit {exit_reason}"
            if pd.notna(exit_price):
                label += f" @ {exit_price:.2f}"
            _register(exit_time, label)

    annotations: List[str] = []
    for ts in candles["timestamp"]:
        notes = events.get(ts, [])
        annotations.append("<br>" + "<br>".join(notes) if notes else "")
    return annotations


def _add_trade_markers(
    fig: go.Figure, trades: pd.DataFrame, *, row: int = 1, col: int = 1
) -> None:
    """Add entry/exit markers with hover suppressed."""

    if trades.empty:
        return
    long_entries = trades[trades["side"] == "LONG"]
    short_entries = trades[trades["side"] == "SHORT"]
    entries = pd.concat([long_entries, short_entries])
    exits = trades.copy()

    if not entries.empty:
        fig.add_trace(
            go.Scatter(
                x=entries["entry_time"],
                y=entries["entry_price"],
                mode="markers",
                name="Entry",
                marker=dict(
                    symbol="triangle-up",
                    size=10,
                    color=[
                        COLOR_LONG if side == "LONG" else COLOR_SHORT
                        for side in entries["side"]
                    ],
                    line=dict(width=1, color="#1f2937"),
                ),
                hoverinfo="skip",
                text=entries["side"],
            ),
            row=row,
            col=col,
        )
    if not exits.empty:
        fig.add_trace(
            go.Scatter(
                x=exits["exit_time"],
                y=exits["exit_price"],
                mode="markers",
                name="Exit",
                marker=dict(
                    symbol="x",
                    size=9,
                    color=COLOR_EXIT,
                    line=dict(width=1, color="#1f2937"),
                ),
                hoverinfo="skip",
                text=exits["exit_reason"].fillna("exit"),
            ),
            row=row,
            col=col,
        )


def _compose_hover_text(frame: pd.DataFrame, events: List[str]) -> List[str]:
    hover_text: List[str] = []
    for ts, open_value, high_value, low_value, close_value, event in zip(
        frame["timestamp"],
        frame["open"],
        frame["high"],
        frame["low"],
        frame["close"],
        events,
    ):
        base = [
            f"Time={pd.to_datetime(ts).strftime('%Y-%m-%d %H:%M')}",
            f"Open={open_value:.2f}",
            f"High={high_value:.2f}",
            f"Low={low_value:.2f}",
            f"Close={close_value:.2f}",
        ]
        if event:
            base.append(event.lstrip("<br>"))
        hover_text.append("<br>".join(base))
    return hover_text


def build_candlestick_figure(
    candles: pd.DataFrame,
    trades: Optional[pd.DataFrame] = None,
    *,
    title: str | None = None,
) -> go.Figure:
    """Single-panel candlestick with optional trade markers."""

    frame = _prepare_candle_frame(candles)
    hover_events = _build_event_strings(frame, trades)
    hover_text = _compose_hover_text(frame, hover_events)

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=frame["timestamp"],
            open=frame["open"],
            high=frame["high"],
            low=frame["low"],
            close=frame["close"],
            name="OHLC",
            showlegend=False,
            hoverinfo="text",
            hovertext=hover_text,
        )
    )
    if trades is not None and not trades.empty:
        _add_trade_markers(fig, trades)
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        spikedistance=-1,
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#e5e7eb",
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikethickness=1,
    )
    fig.update_yaxes(showgrid=True, gridcolor="#e5e7eb")
    return fig


def _estimate_bar_width(trades: pd.DataFrame, candles: pd.DataFrame) -> float:
    """Estimate a reasonable column width for return bars."""

    if trades is not None and not trades.empty:
        durations = trades["exit_time"] - trades["entry_time"]
        durations = durations.dropna()
        if not durations.empty:
            positive = durations[durations > pd.Timedelta(0)]
            if not positive.empty:
                return float(positive.median().total_seconds() * 1000)
    if candles is not None and not candles.empty:
        diffs = candles["timestamp"].sort_values().diff().dropna()
        if not diffs.empty:
            return float(diffs.median().total_seconds() * 1000)
    return 60_000.0


def build_trade_overview_figure(
    candles: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    equity: Optional[pd.DataFrame] = None,
    title: str | None = None,
    show_markers: bool = True,
) -> go.Figure:
    """Two-panel figure: price/equity (top) and returns (bottom)."""

    candle_frame = _prepare_candle_frame(candles)
    hover_events = _build_event_strings(candle_frame, trades)
    hover_text = _compose_hover_text(candle_frame, hover_events)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.74, 0.26],
        vertical_spacing=0.12,
        specs=[[{"secondary_y": True}], [{}]],
    )
    fig.add_trace(
        go.Candlestick(
            x=candle_frame["timestamp"],
            open=candle_frame["open"],
            high=candle_frame["high"],
            low=candle_frame["low"],
            close=candle_frame["close"],
            name="OHLC",
            showlegend=False,
            hoverinfo="text",
            hovertext=hover_text,
        ),
        row=1,
        col=1,
    )
    if equity is not None and not equity.empty:
        equity_sorted = equity.sort_values("timestamp")
        start_window = (
            candle_frame["timestamp"].min() if not candle_frame.empty else None
        )
        first_equity_ts = equity_sorted["timestamp"].iloc[0]
        baseline_ts = first_equity_ts
        if start_window is not None and start_window < first_equity_ts:
            baseline_ts = start_window
        if baseline_ts < first_equity_ts:
            baseline_row = pd.DataFrame(
                {
                    "timestamp": [baseline_ts],
                    "equity": [equity_sorted["equity"].iloc[0]],
                }
            )
            equity_sorted = pd.concat([baseline_row, equity_sorted], ignore_index=True)
        fig.add_trace(
            go.Scatter(
                x=equity_sorted["timestamp"],
                y=equity_sorted["equity"],
                name="Equity",
                mode="lines",
                line=dict(color="#0ea5e9", width=2),
            ),
            row=1,
            col=1,
            secondary_y=True,
        )
    if trades is not None and not trades.empty:
        closed = trades.dropna(subset=["exit_time"]).copy()
        if not closed.empty:
            closed["entry_time"] = pd.to_datetime(
                closed["entry_time"], utc=True, errors="coerce"
            )
            closed["exit_time"] = pd.to_datetime(
                closed["exit_time"], utc=True, errors="coerce"
            )
            if show_markers:
                _add_trade_markers(fig, closed, row=1, col=1)
            candle_width_ms = _estimate_bar_width(None, candle_frame)
            width_default = candle_width_ms * 0.8 if candle_width_ms else 60_000.0
            bar_centers = closed["exit_time"].fillna(closed["entry_time"])
            returns = closed["return"].fillna(0.0)
            widths = closed.get("bar_width_ms")
            if widths is not None:
                widths = widths.fillna(width_default)
                widths = widths.where(widths > 0, width_default)
            else:
                widths = pd.Series(width_default, index=returns.index)
            custom_entry = closed["entry_time"].dt.strftime("%Y-%m-%d %H:%M")
            custom_side = closed.get("side", pd.Series("-", index=closed.index)).astype(
                str
            )
            fig.add_trace(
                go.Bar(
                    x=bar_centers,
                    y=returns,
                    width=widths,
                    marker=dict(
                        color=np.where(returns >= 0, COLOR_LONG, COLOR_SHORT),
                        line=dict(width=0),
                    ),
                    name="Trade Return",
                    hovertemplate=(
                        "Entry=%{customdata[0]}<br>"
                        "Side=%{customdata[1]}<br>"
                        "Exit=%{x|%Y-%m-%d %H:%M}<br>"
                        "Return=%{y:.4f}<extra></extra>"
                    ),
                    customdata=np.column_stack((custom_entry, custom_side)),
                    offsetgroup="trade_returns",
                ),
                row=2,
                col=1,
            )
            fig.add_hline(y=0, line_color="#9ca3af", line_width=1, row=2, col=1)
    fig.update_layout(
        title=title or "",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        barmode="overlay",
        modebar_add=[
            "zoom2d",
            "zoomIn2d",
            "zoomOut2d",
            "autoScale2d",
            "resetScale2d",
            "zoomX",
            "zoomOutX",
            "zoomY",
            "zoomOutY",
        ],
        margin=dict(t=60, b=60, l=60, r=60),
        hovermode="x unified",
        spikedistance=-1,
    )
    fig.update_xaxes(
        rangeslider=dict(visible=True, thickness=0.12, bgcolor="#f8fafc"),
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor="#1f2937",
        spikethickness=1,
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title_text="Time",
        row=2,
        col=1,
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor="#1f2937",
        spikethickness=1,
    )
    fig.update_yaxes(
        domain=[0.45, 1],
        title_text="Price",
        row=1,
        col=1,
        secondary_y=False,
        fixedrange=False,
        showgrid=True,
        gridcolor="#e5e7eb",
    )
    fig.update_yaxes(
        domain=[0.45, 1],
        title_text="Equity",
        row=1,
        col=1,
        secondary_y=True,
        fixedrange=False,
        showgrid=True,
        gridcolor="#e5e7eb",
    )
    fig.update_yaxes(
        domain=[0, 0.28],
        title_text="Return",
        row=2,
        col=1,
        fixedrange=False,
        showgrid=True,
        gridcolor="#e5e7eb",
    )
    return fig


def build_scatter_figure(
    trades: pd.DataFrame, *, title: str | None = None
) -> Optional[go.Figure]:
    """Scatter plot: Holding Time (x) vs Return (y)."""

    if trades is None or trades.empty:
        return None

    df = trades.copy()
    if "holding_mins" not in df.columns or "return" not in df.columns:
        return None
    
    # Ensure numeric
    df["holding_mins"] = pd.to_numeric(df["holding_mins"], errors="coerce")
    df["return"] = pd.to_numeric(df["return"], errors="coerce")
    df = df.dropna(subset=["holding_mins", "return"])
    
    if df.empty:
        return None
    
    fig = go.Figure()
    
    # Separate by Side for coloring
    for side, color in [("LONG", COLOR_LONG), ("SHORT", COLOR_SHORT)]:
        subset = df[df["side"] == side]
        if subset.empty:
            continue
            
        fig.add_trace(
            go.Scatter(
                x=subset["holding_mins"],
                y=subset["return"],
                mode="markers",
                name=side,
                marker=dict(
                    color=color,
                    size=8,
                    line=dict(width=1, color="#1f2937"),
                    opacity=0.8
                ),
                text=subset["exit_reason"],
                hovertemplate=(
                    "Return: %{y:.4f}<br>"
                    "Hold: %{x:.1f}m<br>"
                    "Reason: %{text}<extra></extra>"
                )
            )
        )
        
    fig.update_layout(
        title=title,
        xaxis_title="Holding Time (Minutes)",
        yaxis_title="Return",
        template="plotly_white",
        hovermode="closest",
        margin=dict(t=40, b=40, l=40, r=40),
        height=400,
    )
    
    # Add zero line
    fig.add_hline(y=0, line_color="#9ca3af", line_width=1, line_dash="dash")
    
    return fig


__all__ = [
    "build_candlestick_figure",
    "build_trade_overview_figure",
    "build_scatter_figure",
]
