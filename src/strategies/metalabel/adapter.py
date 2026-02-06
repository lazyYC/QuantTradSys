"""
Meta-Labeling Strategy Adapter.
Implements BaseStrategy interface for integration with training infrastructure.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import optuna
import pandas as pd

from strategies.base import BaseStrategy
from strategies.metalabel.params import (
    PrimarySignalParams, TripleBarrierParams, MetaModelParams
)
from strategies.metalabel.primary import generate_primary_signals, get_signal_bars
from strategies.metalabel.labels import build_labeled_dataset
from strategies.metalabel.features import build_meta_features
from strategies.metalabel.model import train_meta_model, MetaModelResult
from strategies.metalabel.backtest import run_backtest, BacktestResult
from strategies.metalabel.optimization import (
    suggest_primary_params, suggest_barrier_params, suggest_meta_params,
    get_all_optuna_params
)

LOGGER = logging.getLogger(__name__)


class MetaLabelStrategy(BaseStrategy):
    """
    Meta-Labeling Strategy adapter.
    
    Implements Marcos Lopez de Prado's Meta-Labeling approach:
    - Layer 1: High-recall primary signals (RSI, MA Cross, Bollinger)
    - Layer 2: Meta-model filter (LightGBM classifier)
    """
    
    def __init__(self):
        self._feature_cache: Optional[pd.DataFrame] = None
        self._last_signal_df: Optional[pd.DataFrame] = None  # Cache between prepare_data and train
    
    @property
    def can_backtest_dev_data(self) -> bool:
        """Disable dev backtest since random split doesn't apply here."""
        return False
    
    def split_data(
        self,
        data: pd.DataFrame,
        test_days: int = 30,
        valid_days: int = 30,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Time-based split for meta-labeling.
        Uses proper temporal ordering (no look-ahead).
        """
        data = data.sort_values("timestamp").reset_index(drop=True)
        
        # Calculate split points
        bars_per_day = 24 * 60  # 1-minute bars
        test_bars = test_days * bars_per_day
        valid_bars = valid_days * bars_per_day
        
        total = len(data)
        test_start = total - test_bars
        valid_start = test_start - valid_bars
        
        train_df = data.iloc[:valid_start]
        valid_df = data.iloc[valid_start:test_start]
        test_df = data.iloc[test_start:]
        
        LOGGER.info(
            f"Data split: Train={len(train_df)}, Valid={len(valid_df)}, Test={len(test_df)}"
        )
        
        return train_df, valid_df, test_df
    
    def warm_up(self, raw_data: pd.DataFrame) -> None:
        """Pre-compute features for the full dataset."""
        LOGGER.info("Warming up MetaLabel feature cache...")
        
        # Pre-compute meta features with default params
        default_params = MetaModelParams()
        self._feature_cache = build_meta_features(raw_data, default_params)
    
    def build_features(
        self, raw_data: pd.DataFrame, params: Dict[str, Any]
    ) -> pd.DataFrame:
        """Build features for meta-model."""
        meta_params = self._parse_meta_params(params)
        return build_meta_features(raw_data, meta_params)
    
    def prepare_data(
        self, raw_data: pd.DataFrame, params: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Prepare labeled dataset for training.
        Returns (labeled_df, metadata).
        
        Note: We return the labeled_df but also store the full signal_df
        in metadata so train() can build features.
        """
        primary_params = self._parse_primary_params(params)
        barrier_params = self._parse_barrier_params(params)
        
        # Generate signals on raw data
        signal_df = generate_primary_signals(raw_data, primary_params)
        signals_only = get_signal_bars(signal_df)
        
        if len(signals_only) < 50:
            raise ValueError(f"Too few signals ({len(signals_only)})")
        
        # Build labeled dataset
        labeled = build_labeled_dataset(signal_df, signals_only, barrier_params)
        
        if len(labeled) < 20:
            raise ValueError(f"Too few labeled samples ({len(labeled)})")
        
        # Check label balance
        pos_ratio = labeled["meta_label"].mean()
        if pos_ratio < 0.05 or pos_ratio > 0.95:
            raise ValueError(f"Imbalanced labels: pos_ratio={pos_ratio:.2f}")
        
        metadata = {
            "num_signals": len(signals_only),
            "num_labeled": len(labeled),
            "pos_ratio": pos_ratio,
            "signal_df": signal_df,  # Store for feature building in train()
        }
        
        # Cache signal_df for use in train()
        self._last_signal_df = signal_df
        
        LOGGER.info(f"Prepared data: signals={len(signals_only)}, labeled={len(labeled)}, pos_ratio={pos_ratio:.2f}")
        
        return labeled, metadata
    
    def train(
        self,
        train_data: pd.DataFrame,
        valid_data: Optional[pd.DataFrame],
        params: Dict[str, Any],
        model_dir: str,
        seed: int,
    ) -> MetaModelResult:
        """
        Train the meta-model.
        
        Args:
            train_data: Labeled DataFrame from prepare_data
            valid_data: Not used (internal split)
            params: Full params including metadata
            model_dir: Directory to save model
            seed: Random seed
        """
        meta_params = self._parse_meta_params(params)
        
        # Get signal_df from instance cache (set by prepare_data)
        signal_df = self._last_signal_df
        
        if signal_df is None:
            # Fallback: use feature cache from warm_up
            if self._feature_cache is not None:
                LOGGER.warning("No cached signal_df, using feature cache")
                signal_df = self._feature_cache
            else:
                LOGGER.warning("No signal_df or feature cache, using train_data directly")
                signal_df = train_data
        
        # Train meta-model
        result = train_meta_model(
            labeled_df=train_data,
            full_df=signal_df,
            params=meta_params,
            model_dir=model_dir,
            seed=seed,
        )
        
        return result

    
    def get_optuna_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest parameters for Optuna trial."""
        return get_all_optuna_params(trial)
    
    def backtest(
        self,
        raw_data: pd.DataFrame,
        params: Dict[str, Any],
        model_path: Optional[str] = None,
        core_start: Optional[pd.Timestamp] = None,
    ) -> BacktestResult:
        """Run backtest with meta-labeling strategy."""
        primary_params = self._parse_primary_params(params)
        barrier_params = self._parse_barrier_params(params)
        meta_params = self._parse_meta_params(params)
        
        transaction_cost = params.get("transaction_cost", 0.001)
        
        return run_backtest(
            df=raw_data,
            model_path=model_path,
            primary_params=primary_params,
            barrier_params=barrier_params,
            meta_params=meta_params,
            core_start=core_start,
            transaction_cost=transaction_cost,
        )
    
    # === UI Configuration (Plugin Architecture) ===
    
    def get_tunable_params_schema(self) -> Dict[str, Any]:
        """Define UI sliders for interactive backtest."""
        return {
            "meta_threshold": {
                "type": "float",
                "min": 0.3,
                "max": 0.8,
                "step": 0.05,
                "default": 0.5,
                "label": "Meta Threshold",
            },
            "take_profit_pct": {
                "type": "float",
                "min": 0.002,
                "max": 0.015,
                "step": 0.001,
                "default": 0.005,
                "label": "Take Profit %",
            },
            "stop_loss_pct": {
                "type": "float",
                "min": 0.001,
                "max": 0.01,
                "step": 0.001,
                "default": 0.003,
                "label": "Stop Loss %",
            },
            "rsi_oversold": {
                "type": "float",
                "min": 20.0,
                "max": 40.0,
                "step": 2.0,
                "default": 30.0,
                "label": "RSI Oversold",
            },
            "rsi_overbought": {
                "type": "float",
                "min": 60.0,
                "max": 80.0,
                "step": 2.0,
                "default": 70.0,
                "label": "RSI Overbought",
            },
        }
    
    def get_ui_defaults(self) -> Dict[str, Any]:
        """Return default values for UI."""
        schema = self.get_tunable_params_schema()
        return {key: val["default"] for key, val in schema.items()}
    
    def get_template_name(self) -> str:
        """Use generic playground template."""
        return "playground.html"
    
    def get_optimization_metric(self) -> tuple:
        """Meta-labeling optimizes F1 score (balance precision and recall)."""
        return ("f1", "maximize")
    
    # === Parameter Parsing Helpers ===
    
    def _parse_primary_params(self, params: Dict[str, Any]) -> PrimarySignalParams:
        return PrimarySignalParams(
            rsi_window=params.get("rsi_window", 14),
            rsi_oversold=params.get("rsi_oversold", 30.0),
            rsi_overbought=params.get("rsi_overbought", 70.0),
            fast_ma_window=params.get("fast_ma_window", 10),
            slow_ma_window=params.get("slow_ma_window", 30),
            bb_window=params.get("bb_window", 20),
            bb_std=params.get("bb_std", 2.0),
            signal_mode=params.get("signal_mode", "any"),
        )
    
    def _parse_barrier_params(self, params: Dict[str, Any]) -> TripleBarrierParams:
        return TripleBarrierParams(
            take_profit_pct=params.get("take_profit_pct", 0.005),
            stop_loss_pct=params.get("stop_loss_pct", 0.003),
            max_holding_bars=params.get("max_holding_bars", 240),
        )
    
    def _parse_meta_params(self, params: Dict[str, Any]) -> MetaModelParams:
        return MetaModelParams(
            num_leaves=params.get("num_leaves", 31),
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.05),
            n_estimators=params.get("n_estimators", 200),
            min_child_samples=params.get("min_child_samples", 20),
            reg_alpha=params.get("reg_alpha", 0.1),
            reg_lambda=params.get("reg_lambda", 0.1),
            meta_threshold=params.get("meta_threshold", 0.5),
            volatility_window=params.get("volatility_window", 20),
            volume_window=params.get("volume_window", 20),
        )
