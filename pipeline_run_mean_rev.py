import logging
from pathlib import Path

from optimization.mean_reversion_optuna import optimize_mean_reversion
from utils.formatting import format_metrics


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

if __name__ == "__main__":
    result = optimize_mean_reversion(
        symbol="BTC/USDT",
        timeframe="5m",
        lookback_days=400,
        n_trials=100,
        params_store_path=Path("storage/strategy_state.db"),
        trades_store_path=Path("storage/strategy_state.db"),
    )
    print("Best params:", result.train_test.best_params.as_dict(rounded=True))
    print("Train metrics:", format_metrics(result.train_test.train.metrics))
    print("Test metrics:", format_metrics(result.train_test.test.metrics))
