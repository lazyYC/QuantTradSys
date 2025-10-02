"""均值回歸策略專用的報表與圖表工具。"""
from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def rankings_to_dataframe(rankings: Iterable[Mapping[str, object]]) -> pd.DataFrame:
    """將網格搜尋結果轉換為 DataFrame，方便下游統計。"""
    if rankings is None:
        return pd.DataFrame()
    frame = pd.DataFrame(list(rankings))
    if frame.empty:
        return frame
    numeric_cols = [
        "annualized_return",
        "total_return",
        "sharpe",
        "max_drawdown",
        "win_rate",
        "trades",
        "ma_fast",
        "ma_slow",
        "rsi_period",
        "rsi_threshold",
    ]
    for col in numeric_cols:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["avg_trade_return"] = np.where(
        frame.get("trades", 0).fillna(0) > 0,
        frame.get("total_return", 0) / frame.get("trades", 0).replace(0, np.nan),
        np.nan,
    )
    return frame


def _format_series_for_table(series: pd.Series, *, decimals: int = 4) -> tuple[Sequence[str], Sequence[str]]:
    labels: list[str] = []
    values: list[str] = []
    for key, value in series.items():
        if value is None or (isinstance(value, float) and pd.isna(value)):
            display = "-"
        elif isinstance(value, (float, np.floating)):
            display = f"{value:.{decimals}f}"
        else:
            display = str(value)
        labels.append(key)
        values.append(display)
    return labels, values




def _table_layout_settings(row_count: int, *, title_present: bool, min_height: int = 180, max_height: int | None = None) -> dict[str, object]:
    """è¨ç®è¡¨æ ¼é©åçé«åº¦èéè·ã"""
    effective_rows = max(row_count, 1)
    base = 48 if title_present else 32
    height = base + 28 * effective_rows
    if max_height is not None:
        height = min(height, max_height)
    height = max(height, min_height)
    top_margin = 40 if title_present else 16
    return {"height": height, "margin": dict(t=top_margin, b=16, l=20, r=20)}


def create_metrics_table(metrics_df: pd.DataFrame, *, title: str | None = None) -> Optional[go.Figure]:
    """å°ç©æææ¨è½çº Plotly è¡¨æ ¼ã"""
    if metrics_df is None or metrics_df.empty:
        return None
    display_cols = [
        "annualized_return",
        "total_return",
        "sharpe",
        "max_drawdown",
        "win_rate",
        "trades",
    ]
    row = metrics_df.iloc[0]
    subset = row[[col for col in display_cols if col in row.index]].rename({
        "annualized_return": "Annualized Return",
        "total_return": "Total Return",
        "sharpe": "Sharpe",
        "max_drawdown": "Max Drawdown",
        "win_rate": "Win Rate",
        "trades": "Trades",
    })
    labels, values = _format_series_for_table(subset)
    table = go.Table(
        header=dict(values=["Metric", "Value"], fill_color="#0f172a", font=dict(color="white"), align="left", height=34),
        cells=dict(values=[labels, values], fill_color="#f8fafc", align="left", height=28),
    )
    fig = go.Figure(data=[table])
    layout_kwargs = _table_layout_settings(len(labels), title_present=bool(title))
    fig.update_layout(title=title, template="plotly_white", **layout_kwargs)
    return fig



def create_params_table(params: Mapping[str, object] | None, *, title: str | None = None) -> Optional[go.Figure]:
    """å°ç­ç¥åæ¸ä»¥è¡¨æ ¼åç¾ã"""
    if not params:
        return None
    series = pd.Series(params)
    labels, values = _format_series_for_table(series, decimals=3)
    table = go.Table(
        header=dict(values=["Parameter", "Value"], fill_color="#0f172a", font=dict(color="white"), align="left", height=34),
        cells=dict(values=[labels, values], fill_color="#f1f5f9", align="left", height=28),
    )
    fig = go.Figure(data=[table])
    layout_kwargs = _table_layout_settings(len(labels), title_present=bool(title))
    fig.update_layout(title=title, template="plotly_white", **layout_kwargs)
    return fig



def create_trade_distribution_table(trades: pd.DataFrame, *, title: str | None = None) -> Optional[go.Figure]:
    """è¨ç®äº¤æå ±é¬èæææéçæè¦çµ±è¨ã"""
    if trades is None or trades.empty:
        return None
    closed = trades.dropna(subset=["exit_time"]).copy()
    if closed.empty:
        return None
    stats: dict[str, float] = {}
    if "return" in closed.columns:
        returns = pd.to_numeric(closed["return"], errors="coerce").dropna()
        if not returns.empty:
            stats["Return Mean"] = returns.mean()
            stats["Return Median"] = returns.median()
            stats["Return Min"] = returns.min()
            stats["Return Max"] = returns.max()
    if "holding_mins" in closed.columns:
        holding = pd.to_numeric(closed["holding_mins"], errors="coerce").dropna()
        if not holding.empty:
            stats["Holding Mean (m)"] = holding.mean()
            stats["Holding Median (m)"] = holding.median()
            stats["Holding Min (m)"] = holding.min()
            stats["Holding Max (m)"] = holding.max()
    if not stats:
        return None
    labels, values = _format_series_for_table(pd.Series(stats), decimals=4)
    table = go.Table(
        header=dict(values=["Metric", "Value"], fill_color="#0f172a", font=dict(color="white"), align="left", height=34),
        cells=dict(values=[labels, values], fill_color="#e2e8f0", align="left", height=28),
    )
    fig = go.Figure(data=[table])
    layout_kwargs = _table_layout_settings(len(labels), title_present=bool(title))
    fig.update_layout(title=title, template="plotly_white", **layout_kwargs)
    return fig



def create_top_trades_table(trades: pd.DataFrame, *, top_n: int | None = None, title: str | None = None) -> Optional[go.Figure]:
    """ååºå N ç­äº¤æçµæã"""
    if trades is None or trades.empty:
        return None
    closed = trades.dropna(subset=["exit_time"]).copy()
    if closed.empty:
        return None
    closed = closed.sort_values("exit_time", ascending=False)
    if top_n is not None:
        closed = closed.head(top_n)
    closed["return"] = closed["return"].map(lambda v: f"{v:.4f}" if pd.notna(v) else "-")
    closed['entry_time'] = pd.to_datetime(closed['entry_time'], utc=True, errors='coerce')
    closed['exit_time'] = pd.to_datetime(closed['exit_time'], utc=True, errors='coerce')
    closed['entry_time'] = closed['entry_time'].dt.strftime('%Y-%m-%d %H:%M')
    closed['exit_time'] = closed['exit_time'].dt.strftime('%Y-%m-%d %H:%M')
    columns = ["entry_time", "exit_time", "side", "return", "holding_mins", "exit_reason"]
    available_cols = [col for col in columns if col in closed.columns]
    header = [
        {"entry_time": "Entry", "exit_time": "Exit", "side": "Side", "return": "Return", "holding_mins": "Holding (m)", "exit_reason": "Reason"}[col]
        for col in available_cols
    ]
    cell_values = [closed[col].tolist() for col in available_cols]
    table = go.Table(
        header=dict(values=header, fill_color="#0f172a", font=dict(color="white"), align="left", height=34),
        cells=dict(values=cell_values, fill_color="#ffffff", align="left", height=28),
    )
    fig = go.Figure(data=[table])
    layout_kwargs = _table_layout_settings(len(closed), title_present=bool(title), min_height=320, max_height=640)
    fig.update_layout(title=title, template="plotly_white", **layout_kwargs)
    return fig

__all__ = [
    "rankings_to_dataframe",
    "create_params_table",
    "create_metrics_table",
    "create_trade_distribution_table",
    "create_top_trades_table",
]
