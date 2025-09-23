import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import optuna
import pandas as pd
import numpy as np

from data_pipeline.ccxt_fetcher import fetch_yearly_ohlcv
from persistence.param_store import save_strategy_params
from persistence.trade_store import save_trades
from pipelines.mean_reversion import MeanReversionGrid, TrainTestResult, split_train_test
from strategies.data_utils import prepare_ohlcv_frame
from strategies.mean_reversion import (
    MeanReversionFeatureCache,
    MeanReversionParams,
    backtest_mean_reversion,
)

LOGGER = logging.getLogger(__name__)

PERIODS_PER_YEAR = int(365 * 24 * 60 / 5)


@dataclass
class OptimizationResult:
    study: optuna.Study
    train_test: TrainTestResult


def _build_cache(df: pd.DataFrame, grid: MeanReversionGrid) -> MeanReversionFeatureCache:
    return MeanReversionFeatureCache(
        df,
        sma_windows=sorted({int(v) for v in grid.sma_windows}),
        atr_windows=sorted({int(v) for v in grid.atr_windows}),
        volume_windows=sorted({int(v) for v in grid.volume_windows}),
        pattern_windows=sorted({int(v) for v in grid.pattern_mins}),
    )


def _suggest_params(trial: optuna.Trial) -> MeanReversionParams:
    return MeanReversionParams(
        sma_window=trial.suggest_int("sma_window", 20, 80, step=5),
        bb_std=trial.suggest_float("bb_std", 1.5, 3.0, step=0.1),
        atr_window=trial.suggest_int("atr_window", 14, 40, step=2),
        atr_mult=trial.suggest_float("atr_mult", 0.3, 2.0, step=0.1),
        entry_zscore=trial.suggest_float("entry_zscore", 0.8, 3.0, step=0.1),
        volume_window=trial.suggest_int("volume_window", 20, 120, step=10),
        volume_z=trial.suggest_float("volume_z", 0.0, 1.5, step=0.1),
        pattern_min=trial.suggest_int("pattern_min", 1, 5),
        stop_loss_mult=trial.suggest_float("stop_loss_mult", 1.0, 3.0, step=0.1),
        exit_zscore=trial.suggest_float("exit_zscore", 0.0, 1.0, step=0.1),
    )


def _report_pruner(trial: optuna.Trial, equity_df: pd.DataFrame, n_steps: int = 5) -> None:
    if equity_df.empty:
        trial.report(-1.0, step=0)
        return
    total = len(equity_df)
    for step in range(1, n_steps + 1):
        idx = min(int(total * step / n_steps) - 1, total - 1)
        value = float(equity_df["equity"].iloc[idx] - 1)
        trial.report(value, step=step)
        if trial.should_prune():
            raise optuna.TrialPruned()


def optimize_mean_reversion(
    symbol: str,
    *,
    timeframe: str = "5m",
    lookback_days: int = 400,
    n_trials: int = 200,
    timeout: Optional[int] = None,
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    sampler: Optional[optuna.samplers.BaseSampler] = None,
    pruner: Optional[optuna.pruners.BasePruner] = None,
    grid: Optional[MeanReversionGrid] = None,
    exchange_id: str = "binance",
    exchange_config: Optional[dict] = None,
    output_path: Optional[Path] = None,
    params_store_path: Optional[Path] = None,
    trades_store_path: Optional[Path] = None,
) -> OptimizationResult:
    sampler = sampler or optuna.samplers.TPESampler(multivariate=True, group=True)
    pruner = pruner or optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=2)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=bool(storage and study_name),
    )

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
        raise ValueError("Insufficient data for optimization; adjust lookback or timeframe")

    search_grid = grid or MeanReversionGrid(
        sma_windows=range(20, 81, 5),
        bb_stds=[round(x, 1) for x in np.arange(1.5, 3.1, 0.2)],
        atr_windows=range(14, 41, 2),
        atr_mults=[round(x, 1) for x in np.arange(0.5, 2.1, 0.1)],
        entry_zscores=[round(x, 1) for x in np.arange(0.8, 3.1, 0.1)],
        volume_windows=range(20, 121, 10),
        volume_zscores=[round(x, 1) for x in np.arange(0.0, 1.6, 0.1)],
        pattern_mins=range(1, 6),
        stop_loss_mults=[round(x, 1) for x in np.arange(1.0, 3.1, 0.1)],
        exit_zscores=[round(x, 1) for x in np.arange(0.0, 1.1, 0.1)],
    )

    train_cache = _build_cache(train_df, search_grid)
    test_cache = _build_cache(test_df, search_grid)

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial)
        trial.set_user_attr("params", params.__dict__)
        result = backtest_mean_reversion(train_df, params, feature_cache=train_cache)
        equity_df = result.equity_curve
        _report_pruner(trial, equity_df)
        score = result.metrics["annualized_return"] - 0.5 * result.metrics["max_drawdown"]
        return score

    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    best_params_dict = study.best_trial.user_attrs["params"]
    best_params = MeanReversionParams(**{k: int(v) if "window" in k or k == "pattern_min" or k == "atr_window" else float(v) for k, v in best_params_dict.items()})

    train_result = backtest_mean_reversion(train_df, best_params, feature_cache=train_cache)
    test_result = backtest_mean_reversion(test_df, best_params, feature_cache=test_cache)
    train_test = TrainTestResult(best_params=best_params, train=train_result, test=test_result, rankings=[])

    run_id = None
    if params_store_path is not None:
        save_strategy_params(
            params_store_path,
            strategy="mean_reversion_optuna",
            symbol=symbol,
            timeframe=timeframe,
            params=best_params.__dict__,
            metrics={k: float(v) for k, v in train_result.metrics.items()},
        )
        run_id = datetime.now(timezone.utc).isoformat()
    if trades_store_path is not None:
        run_id = save_trades(
            trades_store_path,
            strategy="mean_reversion_optuna",
            dataset="train",
            symbol=symbol,
            timeframe=timeframe,
            trades=train_result.trades,
            metrics=train_result.metrics,
            run_id=run_id,
        )
        save_trades(
            trades_store_path,
            strategy="mean_reversion_optuna",
            dataset="test",
            symbol=symbol,
            timeframe=timeframe,
            trades=test_result.trades,
            metrics=test_result.metrics,
            run_id=run_id,
        )

    return OptimizationResult(study=study, train_test=train_test)


__all__ = ["OptimizationResult", "optimize_mean_reversion"]





