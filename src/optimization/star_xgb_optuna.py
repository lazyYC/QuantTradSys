"""使用 Optuna 對 star_xgb 策略進行超參數搜尋、訓練、回測與儲存。"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

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
from strategies.data_utils import prepare_ohlcv_frame
from strategies.star_xgb.backtest import backtest_star_xgb, StarBacktestResult
from strategies.star_xgb.dataset import (
    TARGET_COLUMN,
    DEFAULT_WARMUP_BARS,
    build_training_dataset,
    prepend_warmup_rows,
    split_train_test,
)
from strategies.star_xgb.features import StarFeatureCache
from strategies.star_xgb.labels import build_label_frame
from strategies.star_xgb.model import StarTrainingResult, train_star_model
from strategies.star_xgb.params import StarIndicatorParams, StarModelParams
from utils.formatting import format_metrics
from utils.symbols import canonicalize_symbol

LOGGER = logging.getLogger(__name__)

RESULT_GUARD_DIR = Path("storage/optuna_result_flags")

TRANSACTION_COST = 0.001
MIN_VALIDATION_DAYS = 30
STOP_LOSS_PCT = 0.005


@dataclass
class StarOptunaResult:
    study: optuna.Study
    best_training_result: StarTrainingResult
    train_backtest: StarBacktestResult
    valid_backtest: StarBacktestResult
    test_backtest: StarBacktestResult


def optimize_star_xgb(
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
) -> StarOptunaResult:
    symbol = canonicalize_symbol(symbol)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=bool(storage and study_name),
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
        raise ValueError("train set 為空，請檢查資料來源")
    if test_df.empty:
        raise ValueError("test set 為空，請檢查資料來源")

    def _objective(trial: Trial) -> float:
        indicator_params = _suggest_indicator(trial)
        model_params = _suggest_model(trial)

        cache = StarFeatureCache(
            train_df,
            trend_windows=[indicator_params.trend_window],
            atr_windows=[indicator_params.atr_window],
            volatility_windows=[indicator_params.volatility_window],
            volume_windows=[indicator_params.volume_window],
            pattern_windows=[indicator_params.pattern_lookback],
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
            raise optuna.TrialPruned("資料集為空或只有單一類別")

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                result = train_star_model(
                    dataset,
                    indicator_params,
                    [model_params],
                    model_dir=Path(tmpdir),
                    valid_days=MIN_VALIDATION_DAYS,
                    transaction_cost=TRANSACTION_COST,
                    min_validation_days=MIN_VALIDATION_DAYS,
                    stop_loss_pct=STOP_LOSS_PCT,
                )
            except ValueError as exc:
                raise optuna.TrialPruned(str(exc)) from exc

        score = float(result.validation_metrics.get("total_return", 0.0))
        trial.set_user_attr("indicator_params", indicator_params.as_dict())
        trial.set_user_attr("model_params", model_params.as_dict())
        trial.set_user_attr("validation_metrics", result.validation_metrics)
        return score

    study.optimize(_objective, n_trials=n_trials, timeout=timeout)

    best_trial = study.best_trial
    best_indicator = StarIndicatorParams(**best_trial.user_attrs["indicator_params"])
    best_model_params = StarModelParams(**best_trial.user_attrs["model_params"])

    # 使用最佳參數重新訓練完整模型並執行回測
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

    training_result = train_star_model(
        train_dataset,
        best_indicator,
        [best_model_params],
        model_dir=model_dir,
        valid_days=0,  # 使用全部訓練資料
        transaction_cost=TRANSACTION_COST,
        min_validation_days=MIN_VALIDATION_DAYS,
        stop_loss_pct=STOP_LOSS_PCT,
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
        transaction_cost=TRANSACTION_COST,
        stop_loss_pct=STOP_LOSS_PCT,
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
        transaction_cost=TRANSACTION_COST,
        stop_loss_pct=STOP_LOSS_PCT,
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
        transaction_cost=TRANSACTION_COST,
        stop_loss_pct=STOP_LOSS_PCT,
    )

    # 儲存結果
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
                params_store_path,
                strategy=strategy_key,
                symbol=symbol,
                timeframe=timeframe,
                params=params_payload,
                metrics=format_metrics(test_bt_result.metrics),
            )

        if trades_store_path:
            save_trades(
                trades_store_path,
                strategy=strategy_key,
                dataset="train",
                symbol=symbol,
                timeframe=timeframe,
                trades=train_bt_result.trades,
                metrics=format_metrics(train_bt_result.metrics),
                run_id=run_id,
            )
            save_trades(
                trades_store_path,
                strategy=strategy_key,
                dataset="valid",
                symbol=symbol,
                timeframe=timeframe,
                trades=valid_bt_result.trades,
                metrics=format_metrics(valid_bt_result.metrics),
                run_id=run_id,
            )
            save_trades(
                trades_store_path,
                strategy=strategy_key,
                dataset="test",
                symbol=symbol,
                timeframe=timeframe,
                trades=test_bt_result.trades,
                metrics=format_metrics(test_bt_result.metrics),
                run_id=run_id,
            )
            prune_strategy_trades(
                db_path=trades_store_path,
                strategy=strategy_key,
                symbol=symbol,
                timeframe=timeframe,
                keep_run_id=run_id,
            )
            prune_strategy_metrics(
                db_path=trades_store_path,
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


def _should_persist_results(study: optuna.Study, guard_dir: Optional[Path]) -> bool:
    if _has_pending_trials(study):
        return False
    if guard_dir is None:
        return True
    return _acquire_result_guard(study.study_name, guard_dir)


def _suggest_indicator(trial: Trial) -> StarIndicatorParams:
    return StarIndicatorParams(
        trend_window=trial.suggest_categorical("trend_window", [45, 60, 75]),
        slope_window=trial.suggest_categorical("slope_window", [5, 10]),
        atr_window=trial.suggest_categorical("atr_window", [14, 21, 28]),
        volatility_window=trial.suggest_categorical("volatility_window", [15, 20, 30]),
        volume_window=trial.suggest_categorical("volume_window", [30, 45, 60]),
        pattern_lookback=trial.suggest_categorical("pattern_lookback", [3, 4, 5]),
        upper_shadow_min=trial.suggest_float("upper_shadow_min", 0.65, 0.9, step=0.05),
        body_ratio_max=trial.suggest_float("body_ratio_max", 0.15, 0.25, step=0.025),
        volume_ratio_max=trial.suggest_float("volume_ratio_max", 0.5, 0.9, step=0.05),
        future_window=trial.suggest_categorical("future_window", [5, 7]),
        future_return_threshold=trial.suggest_float(
            "future_return_threshold", 0.005, 0.02, step=0.0005
        ),
    )


def _suggest_model(trial: Trial) -> StarModelParams:
    return StarModelParams(
        num_leaves=trial.suggest_int("num_leaves", 15, 63, step=8),
        max_depth=trial.suggest_int("max_depth", 3, 6),
        learning_rate=trial.suggest_float("learning_rate", 0.03, 0.2, step=0.01),
        n_estimators=trial.suggest_int("n_estimators", 200, 600, step=50),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 25, step=5),
        subsample=trial.suggest_float("subsample", 0.6, 0.9, step=0.05),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 0.9, step=0.05),
        feature_fraction_bynode=trial.suggest_float(
            "feature_fraction_bynode", 0.6, 0.9, step=0.05
        ),
        lambda_l1=trial.suggest_float("lambda_l1", 0.0, 2.0, step=0.1),
        lambda_l2=trial.suggest_float("lambda_l2", 0.0, 2.0, step=0.1),
        bagging_freq=trial.suggest_int("bagging_freq", 1, 5),
        decision_threshold=trial.suggest_float(
            "decision_threshold", 0.004, 0.007, step=0.0005
        ),
    )
