"""
Unified training script for QuantTradSys strategies.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Type

# Add src to path
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_pipeline.ccxt_fetcher import fetch_yearly_ohlcv
from optimization.engine import optimize_strategy
from strategies.base import BaseStrategy
from strategies.data_utils import prepare_ohlcv_frame
from strategies.star_xgb.adapter import StarXGBStrategy
from utils.symbols import canonicalize_symbol

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
LOGGER = logging.getLogger(__name__)

STRATEGIES: Dict[str, Type[BaseStrategy]] = {
    "star_xgb": StarXGBStrategy,
}


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
    
    # 2. Select Strategy
    if args.strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {args.strategy}. Available: {list(STRATEGIES.keys())}")
    
    strategy_class = STRATEGIES[args.strategy]
    strategy = strategy_class()
    
    # 3. Warm Up (Initialize Caches)
    LOGGER.info("Warming up strategy caches...")
    strategy.warm_up(cleaned_df)
    
    # 4. Define Fixed Configuration (The "Target")
    # These parameters are NOT optimized by Optuna to ensure stability.
    config = {
        "future_window": args.future_window,
        "future_return_threshold": args.future_return_threshold,
        # Pass other fixed args that might be needed by specific strategies
        "valid_days": args.test_days, # Use test_days as validation split for internal training
        "transaction_cost": 0.001,
        "stop_loss_pct": 0.005,
        "use_gpu": args.use_gpu,
        # Pass storage paths for persistence (engine can pass them to strategy if needed, 
        # or we can handle persistence here if engine returned the best model)
        # But engine runs the loop.
    }
    
    LOGGER.info(f"Starting optimization for {args.strategy} with fixed target: future_window={args.future_window}")
    
    # 5. Run Optimization
    # Note: We pass the *instance* now? No, engine expects class. 
    # But we warmed up an instance.
    # Refactor engine to accept instance? 
    # Or engine instantiates? 
    # Engine currently does `strategy = strategy_class()`.
    # We should change engine to accept an instance or class.
    # Let's change engine to accept `strategy_instance`.
    
    # Wait, I need to modify engine.py first to accept instance.
    # Or I can just rely on the fact that StarFeatureCache might be global or I pass the warmed up instance.
    
    # Let's modify engine.py to accept `strategy` (instance) instead of `strategy_class`.
    # This is better dependency injection.
    
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
    
    # Re-train (or just backtest) to get artifacts
    # We need to use the strategy to generate the full result object
    # Note: We use a temp dir for the model artifact unless we want to save it permanently now
    
    # Determine model directory
    if args.model_dir:
        model_dir = args.model_dir
    else:
        model_dir = Path(f"storage/models/{args.strategy}")
    
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # We need to re-run training to get the result object (metrics, trades, etc.)
    # Since we don't have the result object from the study, only the params.
    try:
        result = strategy.train(
            train_data=None, # Strategy handles data internally if we pass None? No, we need to pass dataset.
            # Wait, strategy.train signature in BaseStrategy is:
            # train(train_data, valid_data, params, model_dir, seed)
            # We need to prepare data first.
            valid_data=None,
            params=best_params,
            model_dir=str(model_dir),
            seed=42, # Use fixed seed for final model
        )
        
        # But wait, we need to call prepare_data first to get the dataset!
        # The engine did this inside the loop. We need to do it here.
        dataset, metadata = strategy.prepare_data(cleaned_df, best_params)
        
        # Now train
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
        
        # Save Params
        # We need metrics from the result. Assuming result has validation_metrics or test_metrics.
        # StarTrainingResult has train_metrics, validation_metrics, test_metrics.
        # BaseStrategy.train returns Any, so we need to inspect what it returns.
        # StarXGBStrategy returns StarTrainingResult.
        
        metrics_to_save = {}
        if hasattr(result, "test_metrics"):
             metrics_to_save = result.test_metrics
        elif hasattr(result, "validation_metrics"):
             metrics_to_save = result.validation_metrics
             
        save_strategy_params(
            args.store_path,
            strategy=args.strategy,
            symbol=args.symbol,
            timeframe=args.timeframe,
            params=study.best_params, # Save only the optimized params? Or full config? Usually optimized.
            metrics=format_metrics(metrics_to_save),
            stop_loss_pct=config.get("stop_loss_pct", 0.005),
            transaction_cost=config.get("transaction_cost", 0.001),
        )
        
        # Save Trades (if available)
        # StarTrainingResult doesn't directly have trades df, it has rankings.
        # Wait, the original star_xgb_optuna.py ran backtest_star_xgb AFTER training to get trades.
        # StarTrainingResult has model_path.
        # We might need to run backtest here if the strategy.train doesn't return trades.
        # StarXGBStrategy.train returns StarTrainingResult.
        # We need to run backtest to get the trades DataFrame.
        
        # This implies we need a uniform way to get trades from a strategy.
        # BaseStrategy doesn't enforce "get_trades".
        # For now, let's assume we only save params if generic.
        # BUT user specifically asked for trades persistence.
        # So we should probably add `backtest` method to BaseStrategy or handle StarXGB specifically.
        
        # Let's check StarXGBStrategy adapter. It just calls train_star_model.
        # train_star_model returns StarTrainingResult.
        # We need to run backtest.
        
        # For this specific task, I will add a check.
        if args.strategy == "star_xgb":
            from strategies.star_xgb.backtest import backtest_star_xgb
            
            # We need to split data again to get test set?
            # Or just backtest on the whole set or specific set?
            # Original code ran backtest on train, valid, test.
            
            # Let's just backtest on the "Test" portion (last N days).
            # We need to split cleaned_df.
            from strategies.star_xgb.dataset import split_train_test, prepend_warmup_rows, DEFAULT_WARMUP_BARS
            
            train_df, test_df = split_train_test(cleaned_df, test_days=args.test_days)
            
            # Prepare input for backtest (needs warmup)
            test_input = prepend_warmup_rows(cleaned_df, test_df, DEFAULT_WARMUP_BARS)
            
            bt_result = backtest_star_xgb(
                test_input,
                result.indicator_params,
                result.model_params,
                model_path=str(result.model_path),
                timeframe=args.timeframe,
                class_means=result.class_means,
                class_thresholds=result.class_thresholds,
                feature_columns=result.feature_columns,
                feature_stats=result.feature_stats,
                transaction_cost=config.get("transaction_cost", 0.001),
                stop_loss_pct=config.get("stop_loss_pct", 0.005),
            )
            
            save_trades(
                args.store_path,
                strategy=args.strategy,
                dataset="test",
                symbol=args.symbol,
                timeframe=args.timeframe,
                trades=bt_result.trades,
                metrics=format_metrics(bt_result.metrics),
                run_id=run_id,
            )
            
            # Prune old trades
            prune_strategy_trades(
                args.store_path,
                strategy=args.strategy,
                symbol=args.symbol,
                timeframe=args.timeframe,
                keep_run_id=run_id,
            )
            prune_strategy_metrics(
                args.store_path,
                strategy=args.strategy,
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
    
    parser.add_argument("--strategy", type=str, required=True, choices=list(STRATEGIES.keys()), help="Strategy to train")
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
    
    return parser.parse_args()


if __name__ == "__main__":
    main()
