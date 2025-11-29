"""
Unified training script for QuantTradSys strategies.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Type
from datetime import datetime
import tempfile

import pandas as pd

# Add src to path
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_pipeline.ccxt_fetcher import fetch_yearly_ohlcv
from optimization.engine import optimize_strategy
from strategies.base import BaseStrategy
from strategies.data_utils import prepare_ohlcv_frame
from strategies.loader import load_strategy_class
from utils.symbols import canonicalize_symbol

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
LOGGER = logging.getLogger(__name__)

# STRATEGIES dict is removed

def main() -> None:
    args = _parse_args()
    args.symbol = canonicalize_symbol(args.symbol)

    # 1. Load Data
    LOGGER.info(f"Fetching data for {args.symbol} {args.timeframe}...")
    raw_df = fetch_yearly_ohlcv(
        symbol=args.symbol,
        timeframe=args.timeframe,
        lookback_days=args.lookback_days,
        exchange_id=args.exchange,
        prune_history=False,
    )
    cleaned_df = prepare_ohlcv_frame(raw_df, args.timeframe)
    LOGGER.info(f"Loaded data shape: {cleaned_df.shape}")
    if cleaned_df.empty:
        LOGGER.error("Data is empty! Check symbol or fetcher.")
        return
    
    # 2. Select Strategy (Dynamic)
    try:
        strategy_class = load_strategy_class(args.strategy)
    except ValueError as e:
        LOGGER.error(e)
        return
        
    strategy = strategy_class()
    
    # 3. Warm Up (Initialize Caches)
    LOGGER.info("Warming up strategy caches...")
    strategy.warm_up(cleaned_df)
    
    # 4. Define Fixed Configuration (The "Target")
    config = {
        "future_window": args.future_window,
        "future_return_threshold": args.future_return_threshold,
        "valid_days": args.test_days,
        "transaction_cost": 0.001,
        "stop_loss_pct": 0.005,
        "use_gpu": args.use_gpu,
    }
    
    LOGGER.info(f"Starting optimization for {args.strategy} with fixed target: future_window={args.future_window}")
    
    # 5. Run Optimization
    study = optimize_strategy(
        strategy=strategy,
        raw_data=cleaned_df,
        config=config,
        n_trials=args.n_trials,
        n_seeds=args.n_seeds,
        study_name=args.study_name,
        storage=args.storage,
        seed=42,
    )
    
    # 6. Report Results
    print("\n" + "=" * 80)
    print(" Optimization Finished ".center(80, "="))
    print(f"Best Value: {study.best_value:.4f}")
    print("Best Params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("=" * 80)

    # 7. Persist Best Result
    LOGGER.info("Persisting best result...")
    
    # Merge best params with fixed config
    best_params = {**config, **study.best_params}
    
    # Determine model directory
    if args.model_dir:
        model_dir = args.model_dir
    else:
        # Include study_name in model_dir to avoid collisions
        model_dir = Path(f"storage/models/{args.strategy}/{args.study_name}")
    
    model_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Prepare data
        dataset, metadata = strategy.prepare_data(cleaned_df, best_params)
        
        # Train final model
        result = strategy.train(
            train_data=dataset,
            valid_data=None,
            params=best_params,
            model_dir=str(model_dir),
            seed=42,
        )
        
        # Save to DB
        from persistence.param_store import save_strategy_params
        from persistence.trade_store import save_trades, prune_strategy_trades, prune_strategy_metrics
        from utils.formatting import format_metrics
        from datetime import datetime, timezone
        
        run_id = datetime.now(timezone.utc).isoformat()
        
        # Inject training artifacts into params for backtest and persistence
        if hasattr(result, "feature_stats"):
            best_params["feature_stats"] = result.feature_stats
        if hasattr(result, "feature_columns"):
            best_params["feature_columns"] = result.feature_columns
        if hasattr(result, "class_means"):
            best_params["class_means"] = result.class_means
        if hasattr(result, "class_thresholds"):
            best_params["class_thresholds"] = result.class_thresholds

        # Save Params
        metrics_to_save = {}
        if hasattr(result, "test_metrics"):
             metrics_to_save = result.test_metrics
        elif hasattr(result, "validation_metrics"):
             metrics_to_save = result.validation_metrics
             
        # Get model path if available
        model_path_str = None
        if hasattr(result, "model_path"):
            model_path_str = str(result.model_path)
             
        save_strategy_params(
            args.store_path,
            strategy=args.strategy,
            study=args.study_name,
            symbol=args.symbol,
            timeframe=args.timeframe,
            params=best_params,
            metrics=format_metrics(metrics_to_save),
            model_path=model_path_str,
            stop_loss_pct=config.get("stop_loss_pct", 0.005),
            transaction_cost=config.get("transaction_cost", 0.001),
        )
            
        # Run Backtest on Test Set (or specific split if we had logic for it)
        # For now we run on the whole cleaned_df but we should ideally split it or use the result's test set
        # But BaseStrategy.backtest typically runs on whatever data is passed.
        
        # TODO: If we want strict train/valid/test separation in DB, we should split cleaned_df here
        # based on args.test_days and save them as separate datasets.
        
        # Split data for recording
        # from strategies.data_utils import split_train_test # Unused
        
        # Split data for recording based on user requirement:
        # Test: Last 30 days (Day 331-360)
        # Valid: Previous 30 days (Day 301-330)
        # Train: The rest (Day 1-300)
        
        test_days = args.test_days
        # Assuming valid_days is same as test_days for now, or we could add an arg.
        # The user said "Day 301 ~ Day 330 = valid", which implies valid_days = test_days = 30.
        valid_days = test_days 
        
        max_timestamp = cleaned_df["timestamp"].max()
        test_cutoff = max_timestamp - pd.Timedelta(days=test_days)
        valid_cutoff = test_cutoff - pd.Timedelta(days=valid_days)
        
        test_df = cleaned_df[cleaned_df["timestamp"] > test_cutoff]
        valid_df = cleaned_df[(cleaned_df["timestamp"] > valid_cutoff) & (cleaned_df["timestamp"] <= test_cutoff)]
        train_df = cleaned_df[cleaned_df["timestamp"] <= valid_cutoff]
        
        datasets_to_run = [
            ("train", train_df),
            ("valid", valid_df),
            ("test", test_df),
        ]
        
        for ds_name, ds_data in datasets_to_run:
            if ds_data.empty:
                LOGGER.warning(f"Dataset {ds_name} is empty, skipping backtest for it.")
                continue
                
            bt_result = strategy.backtest(
                raw_data=ds_data,
                params=best_params,
                model_path=model_path_str
            )
            
            if bt_result and hasattr(bt_result, "trades") and not bt_result.trades.empty:
                save_trades(
                    args.store_path,
                    strategy=args.strategy,
                    study=args.study_name,
                    dataset=ds_name,
                    symbol=args.symbol,
                    timeframe=args.timeframe,
                    trades=bt_result.trades,
                    metrics=format_metrics(bt_result.metrics),
                    run_id=run_id,
                )
        
        # Prune old trades (only for this study)
        prune_strategy_trades(
            args.store_path,
            strategy=args.strategy,
            study=args.study_name,
            symbol=args.symbol,
            timeframe=args.timeframe,
            keep_run_id=run_id,
        )
        prune_strategy_metrics(
            args.store_path,
            strategy=args.strategy,
            study=args.study_name,
            symbol=args.symbol,
            timeframe=args.timeframe,
            keep_run_id=run_id,
        )
            
        LOGGER.info(f"Results saved to {args.store_path}")

    except Exception as e:
        LOGGER.error(f"Failed to persist results: {e}")
        import traceback
        traceback.print_exc()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and optimize trading strategies.")
    
    parser.add_argument("--strategy", type=str, required=True, help="Strategy to train (must match folder name in src/strategies/)")
    parser.add_argument("--symbol", type=str, default="BTC/USDT:USDT")
    parser.add_argument("--timeframe", type=str, default="5m")
    parser.add_argument("--lookback-days", type=int, default=360)
    parser.add_argument("--exchange", type=str, default="binanceusdm")
    
    # Target Definition (Fixed)
    parser.add_argument("--future-window", type=int, default=5, help="Fixed prediction horizon (bars)")
    parser.add_argument("--future-return-threshold", type=float, default=0.001, help="Fixed return threshold")
    
    # Optimization Config
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--n-seeds", type=int, default=5, help="Number of seeds per trial for stability")
    parser.add_argument("--study-name", type=str, default="strategy_optimization")
    parser.add_argument("--storage", type=str, default="sqlite:///storage/optuna_studies.db")
    parser.add_argument("--test-days", type=int, default=30, help="Days to use for validation/testing")
    parser.add_argument("--use-gpu", action="store_true")
    
    # Storage Paths (Simplified)
    parser.add_argument("--store-path", type=Path, default=Path("storage/strategy_state.db"), help="Path to SQLite DB for params and trades")
    parser.add_argument("--model-dir", type=Path, default=None, help="Directory to save models (default: storage/models/{strategy})")
    parser.add_argument("--dry-run", action="store_true", help="Run without saving to main DB (uses temp DB)")
    
    args = parser.parse_args()
    
    if args.dry_run:
        # Create a temp file for DB
        temp_dir = Path(tempfile.gettempdir()) / "quant_dry_run"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.store_path = temp_dir / f"dry_run_{timestamp}.db"
        
        if args.model_dir is None:
            args.model_dir = temp_dir / f"models_{timestamp}"
            
        LOGGER.warning(f"DRY RUN MODE: Using temp storage at {args.store_path}")
        LOGGER.warning(f"DRY RUN MODE: Models will be saved to {args.model_dir}")
        
    return args


if __name__ == "__main__":
    main()
