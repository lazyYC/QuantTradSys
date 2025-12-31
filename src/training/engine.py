
from __future__ import annotations

import argparse
import logging
import tempfile
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from datetime import datetime, timezone

import pandas as pd
import optuna

from config.env import load_env
from data_pipeline.ccxt_fetcher import fetch_yearly_ohlcv
from persistence.market_store import MarketDataStore
from optimization.engine import optimize_strategy
from strategies.base import BaseStrategy
from strategies.loader import load_strategy_class
from utils.data_utils import prepare_ohlcv_frame
from utils.symbols import canonicalize_symbol
from utils.formatting import format_metrics
from persistence.param_store import save_strategy_params
from persistence.trade_store import save_trades, prune_strategy_trades, prune_strategy_metrics

LOGGER = logging.getLogger(__name__)

@dataclass
class TrainingContext:
    # Arguments
    strategy_name: str
    symbol: str
    timeframe: str
    lookback_days: int
    exchange: str
    transaction_cost: float
    stop_loss_pct: float
    valid_days: int
    test_days: int
    
    # Target / Fixed Config
    future_window: int
    future_return_threshold: float
    use_gpu: bool
    
    # Optimization Config
    n_trials: int
    n_seeds: int
    study_name: str
    storage: str
    
    # Paths & Flags

    model_dir: Optional[Path]
    dry_run: bool = False
    
    # Runtime State
    cleaned_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    strategy_instance: Optional[BaseStrategy] = None
    study: Optional[optuna.Study] = None
    best_params: Dict[str, Any] = field(default_factory=dict)


class TrainingEngine:
    def __init__(self, context: TrainingContext):
        self.ctx = context
        
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> TrainingEngine:
        # Handle Dry Run Logic for paths

        model_dir = args.model_dir
        
        if args.dry_run:
            temp_dir = Path(tempfile.gettempdir()) / "quant_dry_run"
            temp_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # store_path = temp_dir / f"dry_run_{timestamp}.db" # Unused
            if model_dir is None:
                model_dir = temp_dir / f"models_{timestamp}"
            LOGGER.warning(f"DRY RUN MODE: DB writes effectively disabled or pointing to env DB (Logic updated)")
            LOGGER.warning(f"DRY RUN MODE: Models will be saved to {model_dir}")
        else:
            if model_dir is None:
                model_dir = Path(f"storage/models/{args.strategy}/{args.study_name}")
        
        ctx = TrainingContext(
            strategy_name=args.strategy,
            symbol=canonicalize_symbol(args.symbol),
            timeframe=args.timeframe,
            lookback_days=args.lookback_days,
            exchange=args.exchange,
            transaction_cost=args.transaction_cost,
            stop_loss_pct=args.stop_loss_pct,
            valid_days=args.valid_days,
            test_days=args.test_days,
            future_window=args.future_window,
            future_return_threshold=args.future_return_threshold,
            use_gpu=args.use_gpu,
            n_trials=args.n_trials,
            n_seeds=args.n_seeds,
            study_name=args.study_name,
            storage=args.storage,

            model_dir=model_dir,
            dry_run=args.dry_run
        )
        return cls(ctx)

    def run(self) -> None:
        """Execute the full training pipeline."""
        load_env() 
        
        # 1. Load Data
        self._load_data()
        
        # 2. Warm Up Strategy
        self._initialize_strategy()
        
        # 3. Optimize
        self._run_optimization()
        
        # 4. Train Final Model & Persist
        self._train_and_persist()
        
    def _load_data(self) -> None:
        LOGGER.info(f"Fetching data for {self.ctx.symbol} {self.ctx.timeframe}...")
        raw_df = fetch_yearly_ohlcv(
            symbol=self.ctx.symbol,
            timeframe=self.ctx.timeframe,
            lookback_days=self.ctx.lookback_days,
            exchange_id=self.ctx.exchange,
            prune_history=False,
            market_store=MarketDataStore(),
        )
        self.ctx.cleaned_df = prepare_ohlcv_frame(raw_df, self.ctx.timeframe)
        LOGGER.info(f"Loaded data shape: {self.ctx.cleaned_df.shape}")
        if self.ctx.cleaned_df.empty:
            LOGGER.error("Data is empty! Check symbol or fetcher.")
            sys.exit(1)

    def _initialize_strategy(self) -> None:
        try:
            strategy_cls = load_strategy_class(self.ctx.strategy_name)
        except ValueError as e:
            LOGGER.error(e)
            sys.exit(1)
            
        self.ctx.strategy_instance = strategy_cls()
        LOGGER.info("Warming up strategy caches...")
        self.ctx.strategy_instance.warm_up(self.ctx.cleaned_df)

    def _run_optimization(self) -> None:
        config = {
            "future_window": self.ctx.future_window,
            "future_return_threshold": self.ctx.future_return_threshold,
            "valid_days": self.ctx.valid_days,
            "transaction_cost": self.ctx.transaction_cost,
            "stop_loss_pct": self.ctx.stop_loss_pct,
            "use_gpu": self.ctx.use_gpu,
        }
        
        LOGGER.info(f"Starting optimization for {self.ctx.strategy_name} with config: {config}")
        
        self.ctx.study = optimize_strategy(
            strategy=self.ctx.strategy_instance,
            raw_data=self.ctx.cleaned_df,
            config=config,
            n_trials=self.ctx.n_trials,
            n_seeds=self.ctx.n_seeds,
            study_name=self.ctx.study_name,
            storage=self.ctx.storage,
            seed=42,
        )
        
        self.ctx.best_params = {**config, **self.ctx.study.best_params}
        
        print("\n" + "=" * 80)
        print(" Optimization Finished ".center(80, "="))
        print(f"Best Value: {self.ctx.study.best_value:.4f}")
        print("Best Params:")
        for k, v in self.ctx.study.best_params.items():
            print(f"  {k}: {v}")
        print("=" * 80)

    def _train_and_persist(self) -> None:
        LOGGER.info("Persisting best result...")
        self.ctx.model_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Prepare data
            dataset, metadata = self.ctx.strategy_instance.prepare_data(
                self.ctx.cleaned_df, self.ctx.best_params
            )
            
            # Train final model
            result = self.ctx.strategy_instance.train(
                train_data=dataset,
                valid_data=None,
                params=self.ctx.best_params,
                model_dir=str(self.ctx.model_dir),
                seed=42,
            )
            
            run_id = datetime.now(timezone.utc).isoformat()
            
            # Inject artifacts
            if hasattr(result, "feature_stats"):
                self.ctx.best_params["feature_stats"] = result.feature_stats
            if hasattr(result, "feature_columns"):
                self.ctx.best_params["feature_columns"] = result.feature_columns
            if hasattr(result, "class_means"):
                self.ctx.best_params["class_means"] = result.class_means
            if hasattr(result, "class_thresholds"):
                self.ctx.best_params["class_thresholds"] = result.class_thresholds

            # Metrics
            metrics_to_save = {}
            if hasattr(result, "test_metrics"):
                 metrics_to_save = result.test_metrics
            elif hasattr(result, "validation_metrics"):
                 metrics_to_save = result.validation_metrics
                 
            model_path_str = str(result.model_path) if hasattr(result, "model_path") else None
            
            # Save Params
            save_strategy_params(
                strategy=self.ctx.strategy_name,
                study=self.ctx.study_name,
                symbol=self.ctx.symbol,
                timeframe=self.ctx.timeframe,
                params=self.ctx.best_params,
                metrics=format_metrics(metrics_to_save),
                model_path=model_path_str,
                stop_loss_pct=self.ctx.stop_loss_pct,
                transaction_cost=self.ctx.transaction_cost,
            )
            
            # Run Backtest on Splits
            self._run_backtest_on_splits(run_id, model_path_str)
            
            # Prune old trades
            prune_strategy_trades(
                strategy=self.ctx.strategy_name,
                study=self.ctx.study_name,
                symbol=self.ctx.symbol,
                timeframe=self.ctx.timeframe,
                keep_run_id=run_id,
            )
            prune_strategy_metrics(
                strategy=self.ctx.strategy_name,
                study=self.ctx.study_name,
                symbol=self.ctx.symbol,
                timeframe=self.ctx.timeframe,
                keep_run_id=run_id,
            )
            
            LOGGER.info(f"Results saved to Database (PostgreSQL)")

        except Exception as e:
            LOGGER.exception("Failed to persist results")
            import traceback
            traceback.print_exc()

    def _run_backtest_on_splits(self, run_id: str, model_path_str: str | None) -> None:
        # Split Data
        # Assuming valid_days is same as test_days for now structure
        train_df, valid_df, test_df = self.ctx.strategy_instance.split_data(
            self.ctx.cleaned_df, 
            test_days=self.ctx.test_days, 
            valid_days=self.ctx.valid_days
        )
        
        datasets_to_run = [
            ("train", train_df),
            ("valid", valid_df),
            ("test", test_df),
        ]
        
        for ds_name, ds_data in datasets_to_run:
            if ds_data.empty:
                LOGGER.warning(f"Dataset {ds_name} is empty, skipping backtest for it.")
                continue
                
            bt_result = self.ctx.strategy_instance.backtest(
                raw_data=ds_data,
                params=self.ctx.best_params,
                model_path=model_path_str
            )
            
            if bt_result and hasattr(bt_result, "trades") and not bt_result.trades.empty:
                save_trades(
                    strategy=self.ctx.strategy_name,
                    study=self.ctx.study_name,
                    dataset=ds_name,
                    symbol=self.ctx.symbol,
                    timeframe=self.ctx.timeframe,
                    trades=bt_result.trades,
                    metrics=format_metrics(bt_result.metrics),
                    run_id=run_id,
                )
