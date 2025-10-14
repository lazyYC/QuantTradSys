"""報表使用的 Plotly 圖表工具。"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLOR_LONG = "#16a34a"
COLOR_SHORT = "#dc2626"
COLOR_EXIT = "#f97316"


def _add_trade_markers(fig: go.Figure, trades: pd.DataFrame, *, row: int = 1, col: int = 1) -> None:
    """在圖上加入交易進出場標記。"""

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
                    color=[COLOR_LONG if side == "LONG" else COLOR_SHORT for side in entries["side"]],
                    line=dict(width=1, color="#1f2937"),
                ),
                hovertemplate=(
                    "Entry %{text}<br>Price=%{y:.2f}<br>Time=%{x|%Y-%m-%d %H:%M}"
                ),
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
                marker=dict(symbol="x", size=9, color=COLOR_EXIT, line=dict(width=1, color="#1f2937")),
                hovertemplate=(
                    "Exit (%{text})<br>Price=%{y:.2f}<br>Time=%{x|%Y-%m-%d %H:%M}"
                ),
                text=exits["exit_reason"].fillna("exit"),
            ),
            row=row,
            col=col,
        )


def build_candlestick_figure(
    candles: pd.DataFrame,
    trades: Optional[pd.DataFrame] = None,
    *,
    title: str | None = None,
) -> go.Figure:
    """建立含交易標記的 K 線圖。"""

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=candles["timestamp"],
            open=candles["open"],
            high=candles["high"],
            low=candles["low"],
            close=candles["close"],
            name="OHLC",
            showlegend=False,
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
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb")
    fig.update_yaxes(showgrid=True, gridcolor="#e5e7eb")
    return fig


def _estimate_bar_width(trades: pd.DataFrame, candles: pd.DataFrame) -> float:
    """計算交易報酬條圖的預設寬度（毫秒）。"""

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
    """建立價格、資金曲線與單筆報酬圖。"""

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
            x=candles["timestamp"],
            open=candles["open"],
            high=candles["high"],
            low=candles["low"],
            close=candles["close"],
            name="OHLC",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    if equity is not None and not equity.empty:
        equity_sorted = equity.sort_values("timestamp")
        start_window = candles["timestamp"].min() if candles is not None and not candles.empty else None
        first_equity_ts = equity_sorted["timestamp"].iloc[0]
        baseline_ts = first_equity_ts
        if start_window is not None and start_window < first_equity_ts:
            baseline_ts = start_window
        if baseline_ts < first_equity_ts:
            baseline_row = pd.DataFrame({"timestamp": [baseline_ts], "equity": [equity_sorted["equity"].iloc[0]]})
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
            closed["entry_time"] = pd.to_datetime(closed["entry_time"], utc=True, errors="coerce")
            closed["exit_time"] = pd.to_datetime(closed["exit_time"], utc=True, errors="coerce")
            if show_markers:
                _add_trade_markers(fig, closed, row=1, col=1)
            candle_width_ms = _estimate_bar_width(None, candles)
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
            custom_side = closed.get("side", pd.Series('-', index=closed.index)).astype(str)
            fig.add_trace(
                go.Bar(
                    x=bar_centers,
                    y=returns,
                    width=widths,
                    marker=dict(color=np.where(returns >= 0, COLOR_LONG, COLOR_SHORT), line=dict(width=0)),
                    name="Trade Return",
                    hovertemplate='Entry=%{customdata[0]}<br>Side=%{customdata[1]}<br>Exit=%{x|%Y-%m-%d %H:%M}<br>Return=%{y:.4f}',
                    customdata=np.column_stack((custom_entry, custom_side)),
                    offsetgroup='trade_returns',
                ),
                row=2,
                col=1,
            )
            fig.add_hline(y=0, line_color='#9ca3af', line_width=1, row=2, col=1)
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
        xaxis=dict(
            rangeslider=dict(visible=True, thickness=0.12, bgcolor="#f8fafc"),
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikecolor="#1f2937",
            spikethickness=1,
        ),
        hovermode="x unified",
        spikedistance=-1,
    )
    fig.update_yaxes(domain=[0.45, 1], title_text="Price", row=1, col=1, secondary_y=False, fixedrange=False)
    fig.update_yaxes(domain=[0.45, 1], title_text="Equity", row=1, col=1, secondary_y=True, fixedrange=False)
    fig.update_yaxes(domain=[0, 0.28], title_text="Return", row=2, col=1, fixedrange=False)
    fig.update_xaxes(title_text="Time", row=2, col=1, showspikes=True, spikemode="across", spikesnap="cursor", spikecolor="#1f2937", spikethickness=1)
    return fig


__all__ = [
    "build_candlestick_figure",
    "build_trade_overview_figure",
]
