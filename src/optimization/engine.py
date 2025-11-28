from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Type

import optuna
import pandas as pd

from strategies.base import BaseStrategy

LOGGER = logging.getLogger(__name__)


def optimize_strategy(
    strategy: BaseStrategy,
    raw_data: pd.DataFrame,
    config: Dict[str, Any],
    n_trials: int = 50,
    n_seeds: int = 1,
    study_name: str = "strategy_optimization",
    storage: Optional[str] = None,
    metric: str = "total_return",
    direction: str = "maximize",
    seed: int = 42,
) -> optuna.Study:
    """
    Generic optimization loop for any BaseStrategy.

    Args:
        strategy: The initialized strategy instance (already warmed up).
        raw_data: The raw OHLCV data.
        config: Fixed configuration parameters (e.g., target definition).
        n_trials: Number of Optuna trials.
        n_seeds: Number of seeds to average per trial.
        study_name: Name of the study.
        storage: Database URL for Optuna storage.
        metric: The metric to optimize (must be present in validation_metrics).
        direction: "maximize" or "minimize".
        seed: Random seed.

    Returns:
        The completed Optuna study.
    """
    
    # Strategy is already initialized
    # strategy = strategy_class()
    
    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        load_if_exists=True,
        sampler=sampler,
    )

    def objective(trial: optuna.Trial) -> float:
        # 1. Get optimizable hyperparameters from strategy
        optuna_params = strategy.get_optuna_params(trial)
        
        # 2. Merge with fixed config
        # Config takes precedence? No, usually config is fixed targets, optuna is model params.
        # They shouldn't overlap.
        full_params = {**config, **optuna_params}
        
        # 3. Prepare data (Feature Engineering + Labeling)
        # This might be expensive if features depend on params (like window sizes).
        try:
            dataset, metadata = strategy.prepare_data(raw_data, full_params)
        except ValueError as e:
            # Prune trial if data preparation fails (e.g. not enough data for window)
            raise optuna.TrialPruned(str(e))

        if dataset.empty:
             raise optuna.TrialPruned("Empty dataset after preparation")

    # 4. Train with multiple seeds for stability
        seed_scores = []
        seeds = [seed + i * 100 for i in range(n_seeds)]
        
        for seed_val in seeds:
            with tempfile.TemporaryDirectory() as tmpdir:
                try:
                    result = strategy.train(
                        train_data=dataset,
                        valid_data=None, # Assuming strategy handles split internally for now
                        params=full_params,
                        model_dir=tmpdir,
                        seed=seed_val,
                    )
                except Exception as e:
                    LOGGER.exception("Training failed")
                    raise optuna.TrialPruned(f"Training failed: {e}")
                
                # 5. Extract Metric
                if hasattr(result, "validation_metrics"):
                    metrics = result.validation_metrics
                    score = metrics.get(metric)
                    if score is None:
                        raise optuna.TrialPruned(f"Metric {metric} not found in results")
                    seed_scores.append(score)
                else:
                    raise optuna.TrialPruned("Result object does not have validation_metrics")
        
        if not seed_scores:
            LOGGER.warning(f"Trial {trial.number}: No scores computed (all seeds failed).")
            raise optuna.TrialPruned("No scores computed")
            
        avg_score = sum(seed_scores) / len(seed_scores)
        LOGGER.info(f"Trial {trial.number} finished. Avg {metric}: {avg_score:.4f}. Params: {optuna_params}")
        return avg_score

    study.optimize(objective, n_trials=n_trials)
    return study
