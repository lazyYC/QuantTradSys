"""提供 Plotly 圖表產生函式。"""
from __future__ import annotations

from typing import Optional

import pandas as pd
import numpy as np
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
                    "Entry %{text}<br>Price=%{y:.2f}<br>Time=%{x|%Y-%m-%d %H:%M}"  # noqa: E501
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
                    "Exit (%{text})<br>Price=%{y:.2f}<br>Time=%{x|%Y-%m-%d %H:%M}"  # noqa: E501
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


def prepare_metrics_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """整理績效指標供 UI 顯示。"""

    if metrics.empty:
        return metrics
    display = metrics.copy()
    numeric_cols = [
        "annualized_return",
        "total_return",
        "sharpe",
        "max_drawdown",
        "win_rate",
    ]
    for col in numeric_cols:
        if col in display.columns:
            display[col] = display[col].map(lambda x: round(x, 4) if pd.notna(x) else x)
    display["created_at"] = display["created_at"].dt.tz_convert("UTC").dt.strftime("%Y-%m-%d %H:%M")
    return display


def prepare_trades_table(trades: pd.DataFrame) -> pd.DataFrame:
    """轉換交易資料供 UI 顯示。"""

    if trades.empty:
        return trades
    table = trades.copy()
    table["entry_time"] = table["entry_time"].dt.tz_convert("UTC").dt.strftime("%Y-%m-%d %H:%M")
    table["exit_time"] = table["exit_time"].dt.tz_convert("UTC").dt.strftime("%Y-%m-%d %H:%M")
    table["return"] = table["return"].map(lambda x: round(x, 6) if pd.notna(x) else x)
    table["holding_mins"] = table["holding_mins"].map(lambda x: round(x, 1) if pd.notna(x) else x)
    return table


def placeholder_trade_bar_chart(trades: pd.DataFrame) -> Optional[go.Figure]:
    """未來用於逐筆報酬條圖的預留函式。"""

    if trades.empty:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=trades["return"],
            y=trades.index.astype(str),
            orientation="h",
            marker_color=[COLOR_LONG if r >= 0 else COLOR_SHORT for r in trades["return"]],
        )
    )
    fig.update_layout(
        title="Trade Return Distribution",
        xaxis_title="Return",
        yaxis_title="Trade",
        template="plotly_white",
    )
    return fig


__all__ = [
    "build_candlestick_figure",
    "prepare_metrics_table",
    "prepare_trades_table",
    "placeholder_trade_bar_chart",
    "build_trade_overview_figure",
]



def _estimate_bar_width(trades: pd.DataFrame, candles: pd.DataFrame) -> float:
    """計算 bar 的預設寬度（毫秒），避免零寬度造成繪圖異常。"""
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
    """建立上方 K 線、下方交易報酬條的互動圖。"""
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
        equity_sorted = equity.sort_values('timestamp')
        fig.add_trace(
            go.Scatter(
                x=equity_sorted['timestamp'],
                y=equity_sorted['equity'],
                name='Equity',
                mode='lines',
                line=dict(color='#0ea5e9', width=2),
            ),
            row=1,
            col=1,
            secondary_y=True,
        )
    if trades is not None and not trades.empty:
        closed = trades.dropna(subset=['exit_time']).copy()
        if not closed.empty:
            closed['entry_time'] = pd.to_datetime(closed['entry_time'], utc=True, errors='coerce')
            closed['exit_time'] = pd.to_datetime(closed['exit_time'], utc=True, errors='coerce')
            if show_markers:
                _add_trade_markers(fig, closed, row=1, col=1)
            width_default = _estimate_bar_width(closed, candles)
            bar_centers = closed['exit_time'].fillna(closed['entry_time'])
            returns = closed['return'].fillna(0.0)
            widths = closed.get('bar_width_ms')
            if widths is not None:
                widths = widths.fillna(width_default)
                widths = widths.where(widths > 0, width_default)
            else:
                widths = pd.Series(width_default, index=returns.index)
            custom_entry = closed['entry_time'].dt.strftime('%Y-%m-%d %H:%M')
            custom_side = closed.get('side', pd.Series('-', index=closed.index)).astype(str)
            fig.add_trace(
                go.Bar(
                    x=bar_centers,
                    y=returns,
                    width=widths,
                    marker=dict(color=np.where(returns >= 0, '#16a34a', '#dc2626'), line=dict(width=0)),
                    name='Trade Return',
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
        modebar_add=["zoom2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d", "zoomX", "zoomOutX", "zoomY", "zoomOutY"],
        margin=dict(t=60, b=60, l=60, r=60),
        xaxis=dict(rangeslider=dict(visible=True, thickness=0.12, bgcolor="#f8fafc")),
    )
    fig.update_yaxes(domain=[0.45, 1], title_text="Price", row=1, col=1, secondary_y=False, fixedrange=False)
    fig.update_yaxes(domain=[0.45, 1], title_text="Equity", row=1, col=1, secondary_y=True, fixedrange=False)
    fig.update_yaxes(domain=[0, 0.28], title_text="Return", row=2, col=1, fixedrange=False)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    return fig
