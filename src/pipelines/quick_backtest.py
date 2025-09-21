import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from data_pipeline.ccxt_fetcher import fetch_yearly_ohlcv
from strategies.tech_combo import StrategyParams, generate_realtime_signal, grid_search

LOGGER = logging.getLogger(__name__)


@dataclass
class GridConfig:
    ma_fast: Iterable[int]
    ma_slow: Iterable[int]
    rsi_period: Iterable[int]
    rsi_threshold: Iterable[float]
    top_n: int = 5


DEFAULT_GRID = GridConfig(
    ma_fast=[5, 8, 10],
    ma_slow=[12, 21, 34],
    rsi_period=[14, 21],
    rsi_threshold=[50, 55, 60],
    top_n=5,
)


def quick_backtest(df: pd.DataFrame, grid: GridConfig = DEFAULT_GRID) -> List[dict]:
    """以預設網格快速評估策略績效，回傳績效排名。"""
    results = grid_search(
        df,
        ma_fast_range=grid.ma_fast,
        ma_slow_range=grid.ma_slow,
        rsi_period_range=grid.rsi_period,
        rsi_threshold_range=grid.rsi_threshold,
        top_n=grid.top_n,
    )
    LOGGER.info("Backtest completed. top=%s", results[:1])
    return results


def quick_backtest_and_signal(
    symbol: str,
    *,
    timeframe: str = "5m",
    lookback_days: int = 365,
    grid: GridConfig = DEFAULT_GRID,
    exchange_id: str = "binance",
    exchange_config: Optional[dict] = None,
    output_path: Optional[Path] = None,
) -> dict:
    """抓資料、回測、輸出即時訊號的一條龍流程。"""
    df = fetch_yearly_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        lookback_days=lookback_days,
        exchange_id=exchange_id,
        exchange_config=exchange_config,
        output_path=output_path,
    )
    rankings = quick_backtest(df, grid)
    if not rankings:
        LOGGER.warning("No valid strategy configurations found")
        return {"rankings": [], "signal": "HOLD"}
    best = rankings[0]
    params = StrategyParams(
        ma_fast=int(best["ma_fast"]),
        ma_slow=int(best["ma_slow"]),
        rsi_period=int(best["rsi_period"]),
        rsi_threshold=float(best["rsi_threshold"]),
    )
    decision = generate_realtime_signal(df, params)
    return {"rankings": rankings, "best_params": params, "signal": decision}


__all__ = ["GridConfig", "quick_backtest", "quick_backtest_and_signal"]
