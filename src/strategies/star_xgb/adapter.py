from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Sequence

import optuna
import pandas as pd

from strategies.base import BaseStrategy
from strategies.star_xgb.dataset import (
    DEFAULT_WARMUP_BARS,
    TARGET_COLUMN,
    build_training_dataset,
)
from strategies.star_xgb.features import StarFeatureCache
from strategies.star_xgb.labels import build_label_frame
from strategies.star_xgb.model import train_star_model
from strategies.star_xgb.params import StarIndicatorParams, StarModelParams

LOGGER = logging.getLogger(__name__)


class StarXGBStrategy(BaseStrategy):
    """
    Adapter for the StarXGB strategy.
    """

    def __init__(self):
        self._cache: Optional[StarFeatureCache] = None

    def warm_up(self, raw_data: pd.DataFrame) -> None:
        """
        Initialize the feature cache with all possible window sizes defined in the search space.
        This avoids re-calculating rolling windows for every trial.
        """
        # Hardcoded search space for now, or could be passed in config.
        # These match the ranges in get_optuna_params
        trend_windows = [45, 60, 75]
        slope_windows = [5, 10] # Not used in cache init, but used in build_features
        atr_windows = [14, 21, 28]
        volatility_windows = [15, 20, 30]
        volume_windows = [30, 45, 60]
        pattern_windows = [3, 4, 5]
        
        LOGGER.info("Warming up StarFeatureCache...")
        self._cache = StarFeatureCache(
            raw_data,
            trend_windows=trend_windows,
            atr_windows=atr_windows,
            volatility_windows=volatility_windows,
            volume_windows=volume_windows,
            pattern_windows=pattern_windows,
        )

    def build_features(
        self, raw_data: pd.DataFrame, params: Dict[str, Any]
    ) -> pd.DataFrame:
        indicator_params = self._parse_indicator_params(params)
        
        if self._cache is None:
            # Fallback if warm_up wasn't called (e.g. during single run without optimization)
            LOGGER.warning("Cache not initialized, creating temporary cache.")
            cache = StarFeatureCache(
                raw_data,
                trend_windows=[indicator_params.trend_window],
                atr_windows=[indicator_params.atr_window],
                volatility_windows=[indicator_params.volatility_window],
                volume_windows=[indicator_params.volume_window],
                pattern_windows=[indicator_params.pattern_lookback],
            )
            return cache.build_features(indicator_params)
            
        return self._cache.build_features(indicator_params)

    def prepare_data(
        self, raw_data: pd.DataFrame, params: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        
        indicator_params = self._parse_indicator_params(params)
        
        features = self.build_features(raw_data, params)
        labels, thresholds = build_label_frame(features, indicator_params)
        
        dataset = build_training_dataset(
            features,
            labels,
            class_thresholds=thresholds,
            min_abs_future_return=indicator_params.future_return_threshold,
        )
        
        metadata = {
            "class_thresholds": thresholds,
            "indicator_params": indicator_params, # Store for later use
        }
        
        return dataset, metadata

    def train(
        self,
        train_data: pd.DataFrame,
        valid_data: Optional[pd.DataFrame],
        params: Dict[str, Any],
        model_dir: str,
        seed: int,
    ) -> Any:
        
        indicator_params = self._parse_indicator_params(params)
        model_params = self._parse_model_params(params)
        
        return train_star_model(
            dataset=train_data,
            indicator_params=indicator_params,
            model_candidates=[model_params],
            model_dir=Path(model_dir),
            valid_days=params.get("valid_days", 0),
            transaction_cost=params.get("transaction_cost", 0.001),
            min_validation_days=params.get("min_validation_days", 30),
            stop_loss_pct=params.get("stop_loss_pct", 0.005),
            use_gpu=params.get("use_gpu", False),
            seed=seed,
            deterministic=True,
        )

    def get_optuna_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        # Use shared definition (exclude target params as they are fixed in config)
        ind_params = suggest_indicator_params(trial, future_window_choices=None)
        model_params = suggest_model_params(trial)
        
        # Merge into a single dict for engine.py
        return {**ind_params.as_dict(), **model_params.as_dict()}

    def _parse_indicator_params(self, params: Dict[str, Any]) -> StarIndicatorParams:
        # Filter keys that belong to StarIndicatorParams
        valid_keys = StarIndicatorParams.__annotations__.keys()
        filtered = {k: v for k, v in params.items() if k in valid_keys}
        return StarIndicatorParams(**filtered)

    def _parse_model_params(self, params: Dict[str, Any]) -> StarModelParams:
        # Filter keys that belong to StarModelParams
        valid_keys = StarModelParams.__annotations__.keys()
        filtered = {k: v for k, v in params.items() if k in valid_keys}
        return StarModelParams(**filtered)


    def backtest(
        self, 
        raw_data: pd.DataFrame, 
        params: Dict[str, Any], 
        model_path: Optional[str] = None,
        **kwargs
    ) -> Any:
        from strategies.star_xgb.backtest import backtest_star_xgb
        # from strategies.star_xgb.dataset import split_train_test, prepend_warmup_rows, DEFAULT_WARMUP_BARS
        
        # Parse params
        indicator_params = self._parse_indicator_params(params)
        model_params = self._parse_model_params(params)
        
        if not model_path:
             # In backtesting context (e.g. report.py), we might not have a model if we are just testing logic, 
             # but for StarXGB we need the booster.
             # If None, backtest_star_xgb might fail or return empty.
             pass

        class_means = params.get("class_means")
        class_thresholds = params.get("class_thresholds")
        feature_columns = params.get("feature_columns")
        feature_stats = params.get("feature_stats")
        
        return backtest_star_xgb(
            raw_data,
            indicator_params,
            model_params,
            model_path=model_path,
            timeframe=params.get("timeframe", "5m"),
            class_means=class_means,
            class_thresholds=class_thresholds,
            feature_columns=feature_columns,
            feature_stats=feature_stats,
            transaction_cost=params.get("transaction_cost", 0.001),
            stop_loss_pct=params.get("stop_loss_pct", 0.005),
            core_start=kwargs.get("core_start"),
        )

def suggest_indicator_params(
    trial: optuna.Trial, future_window_choices: Optional[Sequence[int]] = None
) -> StarIndicatorParams:
    """
    Suggest indicator parameters using Optuna.
    If future_window_choices is provided, also suggest future_window and return_threshold.
    """
    params = dict(
        trend_window=trial.suggest_categorical("trend_window", [45, 60, 75]),
        slope_window=trial.suggest_categorical("slope_window", [5, 10]),
        atr_window=trial.suggest_categorical("atr_window", [14, 21, 28]),
        volatility_window=trial.suggest_categorical("volatility_window", [15, 20, 30]),
        volume_window=trial.suggest_categorical("volume_window", [30, 45, 60]),
        pattern_lookback=trial.suggest_categorical("pattern_lookback", [3, 4, 5]),
        upper_shadow_min=trial.suggest_float("upper_shadow_min", 0.65, 0.9, step=0.05),
        body_ratio_max=trial.suggest_float("body_ratio_max", 0.16, 0.22, step=0.02),
        volume_ratio_max=trial.suggest_float("volume_ratio_max", 0.55, 0.8, step=0.05),
    )
    
    if future_window_choices:
        params["future_window"] = trial.suggest_categorical(
            "future_window", sorted(set(int(x) for x in future_window_choices))
        )
        params["future_return_threshold"] = trial.suggest_float(
            "future_return_threshold", 0.0, 0.0001, step=0.0001
        )
    else:
        # Defaults
        params["future_window"] = 5
        params["future_return_threshold"] = 0.001

    return StarIndicatorParams(**params)


def suggest_model_params(trial: optuna.Trial) -> StarModelParams:
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
            "decision_threshold", 0.004, 0.007, step=0.0001
        ),
    )

