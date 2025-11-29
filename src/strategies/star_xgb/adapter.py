from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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
        from .optimization import suggest_indicator_params, suggest_model_params
        
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
        model_path: Optional[str] = None
    ) -> Any:
        from strategies.star_xgb.backtest import backtest_star_xgb
        from strategies.star_xgb.dataset import split_train_test, prepend_warmup_rows, DEFAULT_WARMUP_BARS
        
        # Parse params
        indicator_params = self._parse_indicator_params(params)
        model_params = self._parse_model_params(params)
        
        # We assume raw_data is the full dataset or the test set?
        # Usually backtest is run on a specific set.
        # For simplicity, let's assume raw_data is what we want to backtest on.
        # But backtest_star_xgb expects prepared features or raw data?
        # It expects `test_input` which is raw data with warmup.
        
        # If model_path is not provided, we can't really backtest unless we train on the fly?
        # Or maybe we just return empty?
        if not model_path:
             raise ValueError("model_path is required for StarXGB backtest")
             
        # We need to load metadata to get class_means/thresholds if they are not in params.
        # Usually these are saved in the model directory or passed in params.
        # For now, let's assume they are passed in params or we can't run.
        # But wait, `backtest_star_xgb` needs `class_means`, `class_thresholds`.
        # These come from the training result.
        # If we are running from `report.py`, we might load them from DB (metrics?).
        # Or maybe we should load them from the model artifact?
        
        # Let's assume params contains them if they were saved.
        # If not, we might fail.
        
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
        )
