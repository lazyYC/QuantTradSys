from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import optuna
import pandas as pd

from strategies.base import BaseStrategy
from strategies.playground.dataset import (
    DEFAULT_WARMUP_BARS,
    TARGET_COLUMN,
    build_training_dataset,
)
from strategies.playground.features import StarFeatureCache
from strategies.playground.labels import build_label_frame
from strategies.playground.model import train_star_model
from strategies.playground.params import StarIndicatorParams, StarModelParams

LOGGER = logging.getLogger(__name__)


class PlaygroundStrategy(BaseStrategy):
    """
    Adapter for the Playground strategy (StarXGB with random split).
    """

    def __init__(self):
        self._cache: Optional[StarFeatureCache] = None

    @property
    def can_backtest_dev_data(self) -> bool:
        return False

    def split_data(
        self, 
        data: pd.DataFrame, 
        test_days: int = 30, 
        valid_days: int = 30
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Override default time-based split with random split.
        Ignores days parameters and uses fixed ratios or assumes train.py generic split is insufficient.
        Wait, train.py calls this to get datasets to SAVE.
        The user wants 'train/valid/test' separation.
        
        For playground, we want:
        1. Test Set: Still time-based? Or random? 
           Usually 'Test' in backtesting implies 'Out of Sample' -> Time based is safer.
           But 'Valid' inside Train is random.
           
        But `train.py` logic separates 3 chunks.
        If `Playground` wants random split, it usually means:
        Train + Valid are mixed randomly. Test is held out (Time based).
        
        Let's implement:
        Test = Last 30 days (Time based, for final sanity check).
        Train/Valid = Remaining data shuffles.
        """
        # 1. Isolate Test Set (Time Based - to keep a true OOS)
        if data.empty:
            return data.copy(), data.copy(), data.copy()
            
        max_timestamp = data["timestamp"].max()
        test_cutoff = max_timestamp - pd.Timedelta(days=test_days)
        test_df = data[data["timestamp"] > test_cutoff].copy()
        
        # 2. Remaining is Train + Valid
        remaining_df = data[data["timestamp"] <= test_cutoff].copy()
        
        # 3. Random Split for Train/Valid
        from sklearn.model_selection import train_test_split
        # Approx ratio: valid_days / (total_days - test_days)? 
        # Or just fixed 10%? unique 1/11 used in adapter.train implies ~9%
        
        if remaining_df.empty:
             return remaining_df, remaining_df, test_df
             
        train_ds, valid_ds = train_test_split(
            remaining_df, 
            test_size=0.1,  # Fixed 10% for validation
            random_state=42, 
            shuffle=True
        )
        
        return train_ds.reset_index(drop=True), valid_ds.reset_index(drop=True), test_df.reset_index(drop=True)


    def warm_up(self, raw_data: pd.DataFrame) -> None:
        """
        Initialize the feature cache with all possible window sizes defined in the search space.
        This avoids re-calculating rolling windows for every trial.
        """
        # Hardcoded search space (x5 for 1min timeframe)
        # These match the ranges in optimization.py suggest_indicator_params
        trend_windows = list(range(150, 325, 25))
        slope_windows = list(range(25, 100, 25))
        atr_windows = list(range(70, 175, 35))
        volatility_windows = list(range(75, 175, 25))
        volume_windows = [150, 225, 300]
        pattern_windows = [15, 20, 25]
        
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
        
        # Random Split Logic
        from sklearn.model_selection import train_test_split
        # Note: train_data here is the data passed from engine.py, which excludes the final test set.
        # We split this into train/valid randomly.
        train_ds, valid_ds = train_test_split(
            train_data, 
            test_size=1/11, 
            random_state=seed, 
            shuffle=True
        )
        
        return train_star_model(
            dataset=train_ds,
            indicator_params=indicator_params,
            model_candidates=[model_params],
            model_dir=Path(model_dir),
            valid_dataset=valid_ds,
            valid_days=0, # Disable time-based split
            transaction_cost=params.get("transaction_cost", 0.001),
            min_validation_days=0,
            stop_loss_pct=params.get("stop_loss_pct", 0.005),
            use_gpu=params.get("use_gpu", False),
            seed=seed,
            deterministic=True,
            use_vectorized_metrics=True, # Random split requires vectorized eval
        )

    def get_optuna_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        from .optimization import suggest_indicator_params, suggest_model_params
        
        # Use shared definition (exclude target params as they are fixed in config)
        ind_params = suggest_indicator_params(trial, future_window_choices=None)
        model_params = suggest_model_params(trial)
        
        # Merge into a single dict for engine.py
        return {**ind_params.as_dict(), **model_params.as_dict()}

    def _parse_indicator_params(self, params: Dict[str, Any]) -> StarIndicatorParams:
        # Check for nested indicator params
        source = params.copy()
        if "indicator" in params and isinstance(params["indicator"], dict):
            source.update(params["indicator"])
            
        # Filter keys that belong to StarIndicatorParams
        valid_keys = StarIndicatorParams.__annotations__.keys()
        filtered = {k: v for k, v in source.items() if k in valid_keys}
        return StarIndicatorParams(**filtered)

    def _parse_model_params(self, params: Dict[str, Any]) -> StarModelParams:
        # Check for nested model params
        source = params.copy()
        if "model" in params and isinstance(params["model"], dict):
            source.update(params["model"])
            
        # Filter keys that belong to StarModelParams
        valid_keys = StarModelParams.__annotations__.keys()
        filtered = {k: v for k, v in source.items() if k in valid_keys}
        return StarModelParams(**filtered)

    def backtest(
        self, 
        raw_data: pd.DataFrame, 
        params: Dict[str, Any], 
        model_path: Optional[str] = None,
        core_start: Optional[pd.Timestamp] = None,
    ) -> Any:
        from strategies.playground.backtest import backtest_star_xgb
        from strategies.playground.dataset import split_train_test, prepend_warmup_rows, DEFAULT_WARMUP_BARS
        
        # Enforce default future_return_threshold if 0 (train.py default)
        if params.get("future_return_threshold") == 0:
            params["future_return_threshold"] = 0.001
            
        # Parse params
        indicator_params = self._parse_indicator_params(params)
        model_params = self._parse_model_params(params)
        
        # ... (rest of the setup)
        
        class_means = params.get("class_means")
        class_thresholds = params.get("class_thresholds")
        feature_columns = params.get("feature_columns")
        feature_stats = params.get("feature_stats")
        
        main_result = backtest_star_xgb(
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
            use_vectorized_metrics=False, # Report needs detailed trades
            core_start=core_start,
        )

        return main_result

