import itertools
import logging
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd

from data_pipeline.ccxt_fetcher import fetch_yearly_ohlcv
from persistence.param_store import save_strategy_params
from strategies.data_utils import prepare_ohlcv_frame
from strategies.mean_reversion import (
    BacktestResult,
    MeanReversionParams,
    backtest_mean_reversion,
    grid_search_mean_reversion,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_LOOKBACK_DAYS = 400


@dataclass
class MeanReversionGrid:
    sma_windows: Sequence[int]
    bb_stds: Sequence[float]
    atr_windows: Sequence[int]
    atr_mults: Sequence[float]
    entry_zscores: Sequence[float]
    volume_windows: Sequence[int]
    volume_zscores: Sequence[float]
    pattern_mins: Sequence[int]
    stop_loss_mults: Sequence[float]
    exit_zscores: Sequence[float]


DEFAULT_GRID = MeanReversionGrid(
    sma_windows=[20, 40],
    bb_stds=[2.0, 2.5],
    atr_windows=[14, 28],
    atr_mults=[0.5, 1.0],
    entry_zscores=[1.5, 2.0],
    volume_windows=[40, 80],
    volume_zscores=[0.5, 1.0],
    pattern_mins=[2, 3],
    stop_loss_mults=[1.5, 2.0],
    exit_zscores=[0.0, 0.5],
)


def iter_grid(grid: MeanReversionGrid) -> Iterable[MeanReversionParams]:
    for combo in itertools.product(
        grid.sma_windows,
        grid.bb_stds,
        grid.atr_windows,
        grid.atr_mults,
        grid.entry_zscores,
        grid.volume_windows,
        grid.volume_zscores,
        grid.pattern_mins,
        grid.stop_loss_mults,
        grid.exit_zscores,
    ):
        yield MeanReversionParams(*combo)


@dataclass
class TrainTestResult:
    best_params: MeanReversionParams
    train: BacktestResult
    test: BacktestResult
    rankings: List[dict]


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df, df
    end_time = df["timestamp"].iloc[-1]
    test_start = end_time - pd.DateOffset(months=1)
    train_start = test_start - pd.DateOffset(months=12)
    train_df = df[(df["timestamp"] >= train_start) & (df["timestamp"] < test_start)].reset_index(drop=True)
    test_df = df[df["timestamp"] >= test_start].reset_index(drop=True)
    if train_df.empty or test_df.empty:
        split_idx = max(int(len(df) * 0.8), 1)
        train_df = df.iloc[:split_idx].reset_index(drop=True)
        test_df = df.iloc[split_idx:].reset_index(drop=True)
        LOGGER.warning("Fallback split used due to insufficient 12+1 month history")
    return train_df, test_df


def train_mean_reversion(
    symbol: str,
    *,
    timeframe: str = "5m",
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    grid: MeanReversionGrid = DEFAULT_GRID,
    exchange_id: str = "binance",
    exchange_config: Optional[dict] = None,
    output_path: Optional[Path] = None,
    params_store_path: Optional[Path] = None,
) -> TrainTestResult:
    """抓取資料、執行網格搜尋，並回傳訓練與測試結果。"""
    raw_df = fetch_yearly_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        lookback_days=lookback_days,
        exchange_id=exchange_id,
        exchange_config=exchange_config,
        output_path=output_path,
    )
    cleaned = prepare_ohlcv_frame(raw_df, timeframe)
    train_df, test_df = split_train_test(cleaned)
    if train_df.empty or test_df.empty:
        raise ValueError("Insufficient data after split; please extend lookback or verify data integrity")

    rankings = grid_search_mean_reversion(train_df, iter_grid(grid))[:20]
    if not rankings:
        raise ValueError("Grid search produced no valid parameter sets")
    best = rankings[0]
    best_params: MeanReversionParams = best["params"]

    train_result = backtest_mean_reversion(train_df, best_params)
    test_result = backtest_mean_reversion(test_df, best_params)

    if params_store_path is not None:
        save_strategy_params(
            params_store_path,
            strategy="mean_reversion",
            symbol=symbol,
            timeframe=timeframe,
            params=best_params.__dict__,
            metrics={k: float(v) for k, v in train_result.metrics.items()},
        )

    return TrainTestResult(best_params=best_params, train=train_result, test=test_result, rankings=rankings)


__all__ = [
    "MeanReversionGrid",
    "DEFAULT_GRID",
    "train_mean_reversion",
    "TrainTestResult",
]
