import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from data_pipeline.ccxt_fetcher import fetch_yearly_ohlcv
from persistence.param_store import (
    StrategyRecord,
    load_strategy_params,
    save_strategy_params,
)
from strategies.data_utils import prepare_ohlcv_frame
from strategies.tech_combo import (
    StrategyParams,
    generate_realtime_signal,
    grid_search,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_PARAM_STORE = Path("storage/strategy_state.db")


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


def quick_backtest(
    df: pd.DataFrame,
    grid: GridConfig = DEFAULT_GRID,
    *,
    timeframe: str = "5m",
    clean: bool = True,
) -> List[dict]:
    """以預設網格快速評估策略績效，回傳績效排名。"""
    working_df = prepare_ohlcv_frame(df, timeframe) if clean else df
    results = grid_search(
        working_df,
        ma_fast_range=grid.ma_fast,
        ma_slow_range=grid.ma_slow,
        rsi_period_range=grid.rsi_period,
        rsi_threshold_range=grid.rsi_threshold,
        top_n=grid.top_n,
    )
    LOGGER.info("Backtest completed. top=%s", results[:1])
    return results


def train_and_store_best_params(
    symbol: str,
    *,
    timeframe: str = "5m",
    lookback_days: int = 365,
    grid: GridConfig = DEFAULT_GRID,
    exchange_id: str = "binance",
    exchange_config: Optional[dict] = None,
    output_path: Optional[Path] = None,
    params_store_path: Path = DEFAULT_PARAM_STORE,
) -> dict:
    """跑網格搜尋並將最佳參數儲存至策略參數庫。"""
    df = fetch_yearly_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        lookback_days=lookback_days,
        exchange_id=exchange_id,
        exchange_config=exchange_config,
        output_path=output_path,
    )
    cleaned = prepare_ohlcv_frame(df, timeframe)
    rankings = quick_backtest(cleaned, grid, timeframe=timeframe, clean=False)
    if not rankings:
        LOGGER.warning("No valid strategy configurations found for %s", symbol)
        return {"rankings": []}
    best = rankings[0]
    params = StrategyParams(
        ma_fast=int(best["ma_fast"]),
        ma_slow=int(best["ma_slow"]),
        rsi_period=int(best["rsi_period"]),
        rsi_threshold=float(best["rsi_threshold"]),
    )
    record = save_strategy_params(
        params_store_path,
        strategy="tech_combo",
        symbol=symbol,
        timeframe=timeframe,
        params=params.__dict__,
        metrics=best,
    )
    return {"rankings": rankings, "best_params": params, "record": record}


def quick_backtest_and_signal(
    symbol: str,
    *,
    timeframe: str = "5m",
    lookback_days: int = 365,
    grid: GridConfig = DEFAULT_GRID,
    exchange_id: str = "binance",
    exchange_config: Optional[dict] = None,
    output_path: Optional[Path] = None,
    params_store_path: Optional[Path] = None,
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
    cleaned = prepare_ohlcv_frame(df, timeframe)
    rankings = quick_backtest(cleaned, grid, timeframe=timeframe, clean=False)
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
    if params_store_path is not None:
        save_strategy_params(
            params_store_path,
            strategy="tech_combo",
            symbol=symbol,
            timeframe=timeframe,
            params=params.__dict__,
            metrics=best,
        )
    decision = generate_realtime_signal(cleaned, params)
    return {"rankings": rankings, "best_params": params, "signal": decision}


def run_realtime_cycle(
    symbol: str,
    *,
    timeframe: str = "5m",
    lookback_days: int = 365,
    exchange_id: str = "binance",
    exchange_config: Optional[dict] = None,
    output_path: Optional[Path] = None,
    params_store_path: Path = DEFAULT_PARAM_STORE,
) -> dict:
    """每次增量更新資料後，以最新儲存參數生成即時訊號。"""
    record = load_strategy_params(
        params_store_path,
        strategy="tech_combo",
        symbol=symbol,
        timeframe=timeframe,
    )
    if record is None:
        LOGGER.warning("No stored parameters found for %s | %s", symbol, timeframe)
        return {"signal": "HOLD", "reason": "missing_params"}

    df = fetch_yearly_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        lookback_days=lookback_days,
        exchange_id=exchange_id,
        exchange_config=exchange_config,
        output_path=output_path,
    )
    cleaned = prepare_ohlcv_frame(df, timeframe)
    params = StrategyParams(**record.params)
    decision = generate_realtime_signal(cleaned, params)
    return {
        "signal": decision,
        "params": record.params,
        "metrics": record.metrics,
        "updated_at": record.updated_at,
    }


__all__ = [
    "GridConfig",
    "DEFAULT_GRID",
    "quick_backtest",
    "quick_backtest_and_signal",
    "train_and_store_best_params",
    "run_realtime_cycle",
]
