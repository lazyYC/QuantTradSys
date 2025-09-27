"""Gradio 介面組裝。"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List

import gradio as gr
import pandas as pd

from ui.data_service import (
    CandleQuery,
    TradeQuery,
    fetch_candles,
    fetch_metrics,
    fetch_trades,
    list_strategy_configs,
    parse_timestamp,
)
from ui.plotting import build_candlestick_figure, prepare_metrics_table, prepare_trades_table

DEFAULT_DISPLAY_TIMEFRAMES = ["5m", "15m", "30m", "1h", "4h"]


def _ensure_configs() -> List[dict[str, str]]:
    """取得策略清單，若為空則顯示友善錯誤。"""

    configs = list_strategy_configs()
    if not configs:
        raise RuntimeError("策略資料庫目前沒有任何紀錄，請先完成訓練流程。")
    return configs


def _unique_sorted(values: list[str]) -> list[str]:
    return sorted(set(values))


def _default_datetime_range() -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=7)
    return start.isoformat(), now.isoformat()


def _filter_symbols(configs: List[dict[str, str]], strategy: str) -> list[str]:
    return _unique_sorted([cfg["symbol"] for cfg in configs if cfg["strategy"] == strategy])


def _filter_timeframes(configs: List[dict[str, str]], strategy: str, symbol: str) -> list[str]:
    return _unique_sorted(
        [cfg["timeframe"] for cfg in configs if cfg["strategy"] == strategy and cfg["symbol"] == symbol]
    )


def _load_payload(
    strategy: str,
    symbol: str,
    base_timeframe: str,
    display_timeframe: str,
    dataset: str,
    start: str,
    end: str,
) -> tuple:
    """整合圖表、績效與交易資料。"""

    start_ts = parse_timestamp(start)
    end_ts = parse_timestamp(end)
    query = CandleQuery(
        symbol=symbol,
        base_timeframe=base_timeframe,
        start=start_ts,
        end=end_ts,
        target_timeframe=display_timeframe or base_timeframe,
    )
    candles = fetch_candles(query)
    trade_query = TradeQuery(
        strategy=strategy,
        symbol=symbol,
        timeframe=base_timeframe,
        start=start_ts,
        end=end_ts,
        dataset=None if dataset == "all" else dataset,
    )
    trades = fetch_trades(trade_query)
    metrics = fetch_metrics(strategy, symbol, base_timeframe)

    if candles.empty:
        empty_fig = build_candlestick_figure(pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"]))
        return empty_fig, pd.DataFrame(), pd.DataFrame(), "⚠️ 指定區間無資料，請調整時間範圍。"

    fig = build_candlestick_figure(
        candles,
        trades=trades,
        title=f"{strategy} | {symbol} | {display_timeframe or base_timeframe}",
    )
    metrics_table = prepare_metrics_table(metrics)
    trades_table = prepare_trades_table(trades)
    status = f"載入 {len(candles)} 根 K 線，交易筆數 {len(trades_table)}。"
    return fig, metrics_table, trades_table, status


def build_interface() -> gr.Blocks:
    """產生 Gradio Blocks。"""

    configs = _ensure_configs()
    strategies = _unique_sorted([cfg["strategy"] for cfg in configs])
    default_strategy = strategies[0]
    symbols = _filter_symbols(configs, default_strategy)
    default_symbol = symbols[0]
    timeframes = _filter_timeframes(configs, default_strategy, default_symbol)
    default_timeframe = timeframes[0]
    start_iso, end_iso = _default_datetime_range()

    with gr.Blocks(title="Mean Reversion Dashboard") as demo:
        gr.Markdown("## 均值回歸策略 Dashboard")
        configs_state = gr.State(configs)

        with gr.Row():
            strategy_dd = gr.Dropdown(
                label="策略",
                choices=strategies,
                value=default_strategy,
            )
            symbol_dd = gr.Dropdown(
                label="交易對",
                choices=symbols,
                value=default_symbol,
            )
            timeframe_dd = gr.Dropdown(
                label="原始 timeframe",
                choices=timeframes,
                value=default_timeframe,
            )
            display_tf_dd = gr.Dropdown(
                label="顯示 timeframe",
                choices=DEFAULT_DISPLAY_TIMEFRAMES,
                value=default_timeframe if default_timeframe in DEFAULT_DISPLAY_TIMEFRAMES else "5m",
            )
            dataset_dd = gr.Dropdown(label="資料集", choices=["all", "train", "test"], value="all")

        with gr.Row():
            start_dt = gr.DateTime(label="開始時間 (UTC)", value=start_iso)
            end_dt = gr.DateTime(label="結束時間 (UTC)", value=end_iso)
            load_btn = gr.Button("載入資料", variant="primary")

        status_md = gr.Markdown("準備就緒。")
        chart = gr.Plot(label="K 線圖")
        with gr.Row():
            metrics_df = gr.DataFrame(label="績效摘要", interactive=False)
            trades_df = gr.DataFrame(label="交易紀錄", interactive=False)

        def _on_strategy_change(selected_strategy: str, configs_data: list[dict[str, str]]):
            symbols = _filter_symbols(configs_data, selected_strategy)
            value = symbols[0] if symbols else None
            return gr.Dropdown.update(choices=symbols, value=value)

        def _on_symbol_change(selected_strategy: str, selected_symbol: str, configs_data: list[dict[str, str]]):
            tfs = _filter_timeframes(configs_data, selected_strategy, selected_symbol)
            value = tfs[0] if tfs else None
            return gr.Dropdown.update(choices=tfs, value=value)

        def _load_wrapper(strategy: str, symbol: str, base_tf: str, display_tf: str, dataset: str, start: str, end: str):
            try:
                figure, metrics_table, trades_table, status = _load_payload(
                    strategy,
                    symbol,
                    base_tf,
                    display_tf,
                    dataset,
                    start,
                    end,
                )
            except Exception as exc:  # noqa: BLE001
                return None, pd.DataFrame(), pd.DataFrame(), f"載入失敗：{exc}"
            return figure, metrics_table, trades_table, status

        strategy_dd.change(
            _on_strategy_change,
            inputs=[strategy_dd, configs_state],
            outputs=symbol_dd,
        )
        symbol_dd.change(
            _on_symbol_change,
            inputs=[strategy_dd, symbol_dd, configs_state],
            outputs=timeframe_dd,
        )
        load_btn.click(
            _load_wrapper,
            inputs=[strategy_dd, symbol_dd, timeframe_dd, display_tf_dd, dataset_dd, start_dt, end_dt],
            outputs=[chart, metrics_df, trades_df, status_md],
        )

        gr.Markdown(
            "---\n預留區塊：未來將在此加入『逐筆交易清單 + 報酬條圖』介面。"
        )

    return demo


__all__ = ["build_interface"]
