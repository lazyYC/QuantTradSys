import argparse
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.symbols import canonicalize_symbol

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)


def main() -> None:
    from optimization.mean_reversion_optuna import optimize_mean_reversion
    from utils.formatting import format_metrics

    parser = argparse.ArgumentParser(description="Run mean reversion optimization")
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--lookback-days", type=int, default=400)
    parser.add_argument("--timeframe", type=str, default="5m")
    parser.add_argument("--symbol", type=str, default="BTC/USDT")
    parser.add_argument(
        "--params-store-path", type=Path, default=Path("storage/strategy_state.db")
    )
    parser.add_argument(
        "--trades-store-path", type=Path, default=Path("storage/strategy_state.db")
    )
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument("--storage", type=str, default=None)
    args = parser.parse_args()
    args.symbol = canonicalize_symbol(args.symbol)

    result = optimize_mean_reversion(
        symbol=args.symbol,
        timeframe=args.timeframe,
        lookback_days=args.lookback_days,
        n_trials=args.n_trials,
        params_store_path=args.params_store_path,
        trades_store_path=args.trades_store_path,
        n_jobs=args.n_jobs,
        study_name=args.study_name,
        storage=args.storage,
    )
    print("Best params:", result.train_test.best_params.as_dict(rounded=True))
    print("Train metrics:", format_metrics(result.train_test.train.metrics))
    print("Test metrics:", format_metrics(result.train_test.test.metrics))


if __name__ == "__main__":
    main()
