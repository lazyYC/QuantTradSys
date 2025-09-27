"""提供 Plotly 圖表產生函式。"""
from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go

COLOR_LONG = "#16a34a"
COLOR_SHORT = "#dc2626"
COLOR_EXIT = "#f97316"


def _add_trade_markers(fig: go.Figure, trades: pd.DataFrame) -> None:
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
            )
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
            )
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
]
