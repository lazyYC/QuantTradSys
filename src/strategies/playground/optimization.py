from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import warnings
from optuna.exceptions import ExperimentalWarning

# Suppress Optuna experimental warnings (multivariate, group)
warnings.filterwarnings("ignore", category=ExperimentalWarning)

import optuna
from optuna import Trial
from optuna.trial import TrialState
import pandas as pd

from data_pipeline.ccxt_fetcher import fetch_yearly_ohlcv
from persistence.param_store import save_strategy_params
from persistence.trade_store import (
    prune_strategy_metrics,
    prune_strategy_trades,
    save_trades,
)
from utils.data_utils import prepare_ohlcv_frame
from .backtest import backtest_star_xgb, StarBacktestResult
from .dataset import (
    TARGET_COLUMN,
    DEFAULT_WARMUP_BARS,
    build_training_dataset,
    prepend_warmup_rows,
    split_train_test,
)
from .features import StarFeatureCache
from .labels import build_label_frame
from .model import StarTrainingResult, train_star_model
from .params import StarIndicatorParams, StarModelParams
from utils.formatting import format_metrics
from utils.symbols import canonicalize_symbol

LOGGER = logging.getLogger(__name__)

RESULT_GUARD_DIR = Path("storage/optuna_result_flags")

MIN_VALIDATION_DAYS = 30


from sklearn.model_selection import train_test_split

@dataclass
class StarOptunaResult:
    study: optuna.Study
    best_training_result: StarTrainingResult
    train_backtest: StarBacktestResult
    valid_backtest: StarBacktestResult
    test_backtest: StarBacktestResult


def optimize_playground(
    symbol: str,
    timeframe: str,
    *,
    lookback_days: int = 360,
    test_days: int = 30,
    n_trials: int = 50,
    timeout: Optional[int] = None,
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    model_dir: Path = Path("storage/models/star_xgb"),
    params_store_path: Optional[Path] = Path("storage/strategy_state.db"),
    trades_store_path: Optional[Path] = Path("storage/strategy_state.db"),
    exchange_id: str = "binance",
    exchange_config: Optional[dict] = None,
    result_guard_dir: Optional[Path] = RESULT_GUARD_DIR,
    transaction_cost: float = 0.001,
    stop_loss_pct: float = 0.005,
    future_window_choices: Optional[Sequence[int]] = None,
    use_gpu: bool = False,
    seeds: Optional[Sequence[int]] = None,
) -> StarOptunaResult:
    symbol = canonicalize_symbol(symbol)
    if future_window_choices is None:
        future_window_choices = [5]
    if seeds is None:
        seeds = [88, 188, 288]
    sampler = optuna.samplers.TPESampler(seed=88, multivariate=True, group=True)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=bool(storage and study_name),
        sampler=sampler,
    )

    raw_df = fetch_yearly_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        lookback_days=lookback_days,
        exchange_id=exchange_id,
        exchange_config=exchange_config,
        prune_history=False,
    )
    cleaned = prepare_ohlcv_frame(raw_df, timeframe)
    train_df, test_df = split_train_test(cleaned, test_days=test_days)
    if train_df.empty:
        raise ValueError("train set is empty")
    if test_df.empty:
        raise ValueError("test set is empty")

    def _objective(trial: Trial) -> float:
        indicator_params = suggest_indicator_params(trial, future_window_choices)
        model_params = suggest_model_params(trial)

        cache = StarFeatureCache(
            train_df,
            trend_windows=list(range(30, 65, 5)),
            atr_windows=list(range(14, 35, 7)),
            volatility_windows=list(range(15, 35, 5)),
            volume_windows=[30, 45, 60],
            pattern_windows=[3, 4, 5],
        )
        features = cache.build_features(indicator_params)
        labels, thresholds = build_label_frame(features, indicator_params)
        dataset = build_training_dataset(
            features,
            labels,
            class_thresholds=thresholds,
            min_abs_future_return=indicator_params.future_return_threshold,
        )

        if dataset.empty or dataset[TARGET_COLUMN].nunique() < 2:
            raise optuna.TrialPruned("Dataset empty or single class")

        seed_scores: list[float] = []
        best_seed_result: StarTrainingResult | None = None
        best_seed: int | None = None

        for seed_val in seeds:
            # Random Split: 1/11 for validation, 10/11 for training
            train_ds, valid_ds = train_test_split(
                dataset, test_size=1/11, random_state=seed_val, shuffle=True
            )
            
            with tempfile.TemporaryDirectory() as tmpdir:
                try:
                    result = train_star_model(
                        dataset=train_ds,
                        indicator_params=indicator_params,
                        model_candidates=[model_params],
                        model_dir=Path(tmpdir),
                        valid_dataset=valid_ds,
                        valid_days=0, # Disable time-based split
                        transaction_cost=transaction_cost,
                        min_validation_days=0,
                        stop_loss_pct=stop_loss_pct,
                        use_gpu=use_gpu,
                        seed=seed_val,
                        deterministic=True,
                        use_vectorized_metrics=True, # Random split requires vectorized eval
                    )
                except ValueError as exc:
                    raise optuna.TrialPruned(str(exc)) from exc

            val_score = float(result.validation_metrics.get("total_return", 0.0))
            seed_scores.append(val_score)
            if best_seed_result is None or val_score > float(
                best_seed_result.validation_metrics.get("total_return", -1e9)
            ):
                best_seed_result = result
                best_seed = seed_val

        if not seed_scores:
            raise optuna.TrialPruned("no seed scores computed")

        score = float(sum(seed_scores) / len(seed_scores))
        trial.set_user_attr("indicator_params", indicator_params.as_dict())
        trial.set_user_attr("model_params", model_params.as_dict())
        trial.set_user_attr(
            "best_seed_validation_metrics",
            best_seed_result.validation_metrics if best_seed_result else {},
        )
        trial.set_user_attr("best_seed", best_seed)
        return score

    study.optimize(_objective, n_trials=n_trials, timeout=timeout)

    best_trial = study.best_trial
    best_indicator = StarIndicatorParams(**best_trial.user_attrs["indicator_params"])
    best_model_params = StarModelParams(**best_trial.user_attrs["model_params"])

    # Re-train with best params
    train_cache = StarFeatureCache(
        train_df,
        trend_windows=[best_indicator.trend_window],
        atr_windows=[best_indicator.atr_window],
        volatility_windows=[best_indicator.volatility_window],
        volume_windows=[best_indicator.volume_window],
        pattern_windows=[best_indicator.pattern_lookback],
    )
    train_features = train_cache.build_features(best_indicator)
    train_labels, train_thresholds = build_label_frame(train_features, best_indicator)
    train_dataset = build_training_dataset(
        train_features,
        train_labels,
        class_thresholds=train_thresholds,
        min_abs_future_return=best_indicator.future_return_threshold,
    )

    best_seed = best_trial.user_attrs.get("best_seed", seeds[0])

    training_result = train_star_model(
        train_dataset,
        best_indicator,
        [best_model_params],
        model_dir=model_dir,
        valid_days=0,
        transaction_cost=transaction_cost,
        min_validation_days=MIN_VALIDATION_DAYS,
        stop_loss_pct=stop_loss_pct,
        use_gpu=use_gpu,
        seed=best_seed,
        deterministic=True,
    )

    inner_validation_days = max(
        MIN_VALIDATION_DAYS,
        min(test_days, lookback_days - test_days)
        if lookback_days > test_days
        else MIN_VALIDATION_DAYS,
    )
    train_core_df, validation_df = split_train_test(
        train_df, test_days=inner_validation_days
    )
    if train_core_df.empty:
        train_core_df = train_df.copy()
    if validation_df.empty:
        validation_df = pd.DataFrame(columns=train_df.columns)

    warmup_bars = DEFAULT_WARMUP_BARS
    train_core_start = (
        pd.to_datetime(train_core_df["timestamp"], utc=True, errors="coerce").min()
        if not train_core_df.empty
        else None
    )
    if train_core_start is not None and pd.isna(train_core_start):
        train_core_start = None
    validation_start = (
        pd.to_datetime(validation_df["timestamp"], utc=True, errors="coerce").min()
        if not validation_df.empty
        else None
    )
    if validation_start is not None and pd.isna(validation_start):
        validation_start = None
    test_start = (
        pd.to_datetime(test_df["timestamp"], utc=True, errors="coerce").min()
        if not test_df.empty
        else None
    )
    if test_start is not None and pd.isna(test_start):
        test_start = None

    train_core_input = (
        prepend_warmup_rows(train_df, train_core_df, warmup_bars)
        if train_core_start is not None
        else train_core_df
    )
    validation_input = (
        prepend_warmup_rows(train_df, validation_df, warmup_bars)
        if validation_start is not None
        else validation_df
    )
    test_input = (
        prepend_warmup_rows(cleaned, test_df, warmup_bars)
        if test_start is not None
        else test_df
    )

    train_bt_result = backtest_star_xgb(
        train_core_input,
        training_result.indicator_params,
        training_result.model_params,
        model_path=str(training_result.model_path),
        timeframe=timeframe,
        class_means=training_result.class_means,
        class_thresholds=training_result.class_thresholds,
        feature_columns=training_result.feature_columns,
        feature_stats=training_result.feature_stats,
        core_start=train_core_start,
        transaction_cost=transaction_cost,
        stop_loss_pct=stop_loss_pct,
        use_vectorized_metrics=True, # Random split
    )

    valid_bt_result = backtest_star_xgb(
        validation_input,
        training_result.indicator_params,
        training_result.model_params,
        model_path=str(training_result.model_path),
        timeframe=timeframe,
        class_means=training_result.class_means,
        class_thresholds=training_result.class_thresholds,
        feature_columns=training_result.feature_columns,
        feature_stats=training_result.feature_stats,
        core_start=validation_start,
        transaction_cost=transaction_cost,
        stop_loss_pct=stop_loss_pct,
        use_vectorized_metrics=True, # Random split
    )

    test_bt_result = backtest_star_xgb(
        test_input,
        training_result.indicator_params,
        training_result.model_params,
        model_path=str(training_result.model_path),
        timeframe=timeframe,
        class_means=training_result.class_means,
        class_thresholds=training_result.class_thresholds,
        feature_columns=training_result.feature_columns,
        feature_stats=training_result.feature_stats,
        core_start=test_start,
        transaction_cost=transaction_cost,
        stop_loss_pct=stop_loss_pct,
    )

    # Persist results
    if _should_persist_results(study, result_guard_dir):
        run_id = datetime.now(timezone.utc).isoformat()
        strategy_key = study.study_name or "star_xgb_default"
        if params_store_path:
            params_payload = {
                "indicator": training_result.indicator_params.as_dict(rounded=True),
                "model": training_result.model_params.as_dict(rounded=True),
                "model_path": str(training_result.model_path),
                "feature_columns": training_result.feature_columns,
                "class_means": training_result.class_means,
                "class_thresholds": training_result.class_thresholds,
                "feature_stats": training_result.feature_stats,
            }
            save_strategy_params(
                strategy=strategy_key,
                symbol=symbol,
                timeframe=timeframe,
                params=params_payload,
                metrics=format_metrics(test_bt_result.metrics),
                stop_loss_pct=stop_loss_pct,
                transaction_cost=transaction_cost,
            )

        if trades_store_path:
            save_trades(
                strategy=strategy_key,
                dataset="train",
                symbol=symbol,
                timeframe=timeframe,
                trades=train_bt_result.trades,
                metrics=format_metrics(train_bt_result.metrics),
                run_id=run_id,
            )
            save_trades(
                strategy=strategy_key,
                dataset="valid",
                symbol=symbol,
                timeframe=timeframe,
                trades=valid_bt_result.trades,
                metrics=format_metrics(valid_bt_result.metrics),
                run_id=run_id,
            )
            save_trades(
                strategy=strategy_key,
                dataset="test",
                symbol=symbol,
                timeframe=timeframe,
                trades=test_bt_result.trades,
                metrics=format_metrics(test_bt_result.metrics),
                run_id=run_id,
            )
            prune_strategy_trades(
                strategy=strategy_key,
                symbol=symbol,
                timeframe=timeframe,
                keep_run_id=run_id,
            )
            prune_strategy_metrics(
                strategy=strategy_key,
                symbol=symbol,
                timeframe=timeframe,
                keep_run_id=run_id,
            )

    return StarOptunaResult(
        study=study,
        best_training_result=training_result,
        train_backtest=train_bt_result,
        valid_backtest=valid_bt_result,
        test_backtest=test_bt_result,
    )


def _should_persist_results(study: optuna.Study, guard_dir: Optional[Path]) -> bool:
    if _has_pending_trials(study):
        return False
    if guard_dir is None:
        return True
    return _acquire_result_guard(study.study_name, guard_dir)


def _has_pending_trials(study: optuna.Study) -> bool:
    try:
        pending = study.get_trials(
            deepcopy=False, states=(TrialState.RUNNING, TrialState.WAITING)
        )
    except Exception:
        return True
    return bool(pending)


def _acquire_result_guard(study_name: Optional[str], guard_dir: Path) -> bool:
    if not study_name:
        return True
    guard_dir.mkdir(parents=True, exist_ok=True)
    guard_path = guard_dir / f"{study_name}.flag"
    try:
        fd = os.open(guard_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(datetime.now(timezone.utc).isoformat())
    return True


def suggest_indicator_params(
    trial: Trial, future_window_choices: Optional[Sequence[int]] = None
) -> StarIndicatorParams:
    """
    Suggest indicator parameters using Optuna.
    If future_window_choices is provided, also suggest future_window and return_threshold (Legacy mode).
    """
    params = dict(
        trend_window=trial.suggest_int("trend_window", 30, 60, step=5),
        slope_window=trial.suggest_int("slope_window", 5, 15, step=5),
        atr_window=trial.suggest_int("atr_window", 14, 28, step=7),
        volatility_window=trial.suggest_int("volatility_window", 15, 30, step=5),
        volume_window=trial.suggest_categorical("volume_window", [30, 45, 60]),
        pattern_lookback=trial.suggest_categorical("pattern_lookback", [3, 4, 5]),
        upper_shadow_min=trial.suggest_float("upper_shadow_min", 0.65, 0.9, step=0.05),
        body_ratio_max=trial.suggest_float("body_ratio_max", 0.16, 0.22, step=0.02),
        volume_ratio_max=trial.suggest_float("volume_ratio_max", 0.55, 0.8, step=0.05),
    )
    
    # Optimize Target Parameters
    # future_window: 1h to 4h (12 to 48 bars of 5m)
    params["future_window"] = trial.suggest_int("future_window", 12, 60, step=12)
    
    # future_return_threshold: 0.1% to 0.5% (Relaxed from 0.01)
    # Since we use Dynamic Target, this is the MINIMUM distance to consider.
    params["future_return_threshold"] = trial.suggest_float(
        "future_return_threshold", 0.005, 0.01, step=0.001
    )

    # Fixed Parameters for Momentum/Volatility
    # We inject them here so they are carried over to the engine's params
    params.update({
        "rsi_window": 14,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "bb_window": 20,
        "bb_std": 2.0,
        "max_open_trades": 25,  # Increased for deep grid
        "max_global_drawdown_pct": 0.10, # Relaxed for grid holding (10%)
        "require_candle_confirmation": False, 
        # Grid Params (Optimized)
    })
    
    # Grid Optimization
    params["grid_step_atr"] = trial.suggest_float("grid_step_atr", 0.3, 1.5, step=0.1)
    params["max_grid_layers"] = trial.suggest_int("max_grid_layers", 5, 20, step=5)
    params["eject_threshold"] = trial.suggest_float("eject_threshold", 0.35, 0.48, step=0.01)
    
    # Optimize ADX Threshold - Stricter Range (Avoid High ADX entirely)
    params["adx_threshold"] = trial.suggest_int("adx_threshold", 40, 60, step=5) # Grid needs room to breathe, so allow higher volatility? No, we eject on High Trend. So Threshold is "Safe Zone".
    # User said: "Eject if ADX > 50". So safe zone is < 50.
    # Let's optimize the Cutoff.

    return StarIndicatorParams(**params)


def suggest_model_params(trial: Trial) -> StarModelParams:
    return StarModelParams(
        num_leaves=trial.suggest_int("num_leaves", 15, 63, step=8),
        max_depth=trial.suggest_int("max_depth", 3, 6),
        learning_rate=trial.suggest_float("learning_rate", 0.03, 0.2, step=0.01),
        n_estimators=trial.suggest_int("n_estimators", 200, 600, step=50),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 25, step=5),
        subsample=trial.suggest_float("subsample", 0.6, 0.9, step=0.05),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 0.9, step=0.05),
        feature_fraction_bynode=trial.suggest_float("feature_fraction_bynode", 0.6, 0.9, step=0.05),
        lambda_l1=trial.suggest_float("lambda_l1", 0.0, 2.0, step=0.1),
        lambda_l2=trial.suggest_float("lambda_l2", 0.0, 2.0, step=0.1),
        bagging_freq=trial.suggest_int("bagging_freq", 1, 5),
        decision_threshold=trial.suggest_float("decision_threshold", 0.0001, 0.003, step=0.0001),
    )
