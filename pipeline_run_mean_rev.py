import logging
from pathlib import Path

from pipelines.mean_reversion import train_mean_reversion

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

if __name__ == "__main__":
    result = train_mean_reversion(
        symbol="BTC/USDT",
        timeframe="5m",
        lookback_days=400,
        params_store_path=Path("storage/strategy_state.db"),
        trades_store_path=Path("storage/strategy_state.db"),
    )
    print("Best params:", result.best_params)
    print("Train metrics:", result.train.metrics)
    print("Test metrics:", result.test.metrics)
