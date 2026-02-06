"""
Optuna optimization for Meta-Labeling Strategy.
"""
import logging
from typing import Any, Dict

import optuna
import pandas as pd

from strategies.metalabel.params import (
    PrimarySignalParams, TripleBarrierParams, MetaModelParams
)
from strategies.metalabel.primary import generate_primary_signals, get_signal_bars
from strategies.metalabel.labels import build_labeled_dataset
from strategies.metalabel.model import train_meta_model

LOGGER = logging.getLogger(__name__)


def suggest_primary_params(trial: optuna.Trial) -> PrimarySignalParams:
    """Suggest primary signal parameters."""
    return PrimarySignalParams(
        rsi_window=trial.suggest_int("rsi_window", 10, 20),
        rsi_oversold=trial.suggest_float("rsi_oversold", 20, 35),
        rsi_overbought=trial.suggest_float("rsi_overbought", 65, 80),
        fast_ma_window=trial.suggest_int("fast_ma_window", 5, 15),
        slow_ma_window=trial.suggest_int("slow_ma_window", 20, 50),
        bb_window=trial.suggest_int("bb_window", 15, 30),
        bb_std=trial.suggest_float("bb_std", 1.5, 2.5, step=0.1),
        signal_mode="any",  # Keep high recall for Layer 1
    )


def suggest_barrier_params(trial: optuna.Trial) -> TripleBarrierParams:
    """Suggest triple barrier parameters."""
    return TripleBarrierParams(
        take_profit_pct=trial.suggest_float("take_profit_pct", 0.003, 0.01, step=0.001),
        stop_loss_pct=trial.suggest_float("stop_loss_pct", 0.002, 0.007, step=0.001),
        max_holding_bars=trial.suggest_int("max_holding_bars", 120, 480, step=60),
    )


def suggest_meta_params(trial: optuna.Trial) -> MetaModelParams:
    """Suggest meta-model parameters."""
    return MetaModelParams(
        num_leaves=trial.suggest_int("num_leaves", 15, 63),
        max_depth=trial.suggest_int("max_depth", 4, 8),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        n_estimators=trial.suggest_int("n_estimators", 100, 400, step=50),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 50),
        reg_alpha=trial.suggest_float("reg_alpha", 0.01, 1.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 0.01, 1.0, log=True),
        meta_threshold=trial.suggest_float("meta_threshold", 0.4, 0.7, step=0.05),
        volatility_window=20,
        volume_window=20,
    )


def objective(
    trial: optuna.Trial,
    raw_data: pd.DataFrame,
    config: Dict[str, Any],
    model_dir: str,
) -> float:
    """
    Optuna objective for meta-labeling strategy.
    
    Optimization target: F1 score on validation set 
    (balance between precision and recall).
    """
    # Suggest parameters
    primary_params = suggest_primary_params(trial)
    barrier_params = suggest_barrier_params(trial)
    meta_params = suggest_meta_params(trial)
    
    # Generate primary signals
    signal_df = generate_primary_signals(raw_data, primary_params)
    
    # Get signal bars only
    signals_only = get_signal_bars(signal_df)
    
    if len(signals_only) < 100:
        LOGGER.warning(f"Too few signals ({len(signals_only)}), pruning trial")
        raise optuna.TrialPruned()
    
    # Build labeled dataset
    labeled = build_labeled_dataset(signal_df, signals_only, barrier_params)
    
    if len(labeled) < 50:
        LOGGER.warning(f"Too few labeled samples ({len(labeled)}), pruning trial")
        raise optuna.TrialPruned()
    
    # Check label balance
    pos_ratio = labeled["meta_label"].mean()
    if pos_ratio < 0.1 or pos_ratio > 0.9:
        LOGGER.warning(f"Imbalanced labels (pos_ratio={pos_ratio:.2f}), pruning trial")
        raise optuna.TrialPruned()
    
    # Train meta-model
    try:
        result = train_meta_model(
            labeled_df=labeled,
            full_df=signal_df,
            params=meta_params,
            model_dir=f"{model_dir}/trial_{trial.number}",
            seed=42,
        )
    except Exception as e:
        LOGGER.error(f"Training failed: {e}")
        raise optuna.TrialPruned()
    
    # Return F1 score as objective
    f1 = result.validation_metrics.get("f1", 0)
    
    # Store additional metrics for analysis
    trial.set_user_attr("accuracy", result.validation_metrics.get("accuracy", 0))
    trial.set_user_attr("precision", result.validation_metrics.get("precision", 0))
    trial.set_user_attr("recall", result.validation_metrics.get("recall", 0))
    trial.set_user_attr("auc", result.validation_metrics.get("auc", 0))
    trial.set_user_attr("num_signals", len(signals_only))
    trial.set_user_attr("num_labeled", len(labeled))
    
    return f1


def get_all_optuna_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Get all suggested parameters as a flat dictionary."""
    primary = suggest_primary_params(trial)
    barrier = suggest_barrier_params(trial)
    meta = suggest_meta_params(trial)
    
    return {
        **primary.as_dict(),
        **barrier.as_dict(),
        **meta.as_dict(),
    }
