"""star_xgb 策略訓練 CLI."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

EXPLANATIONS: Dict[str, str] = {
    "accuracy": "模型預測K棒漲跌類別的準確率。",
    "mae_expected": "預期報酬的平均絕對誤差，越小代表模型對報酬的預測越準。",
    "threshold": "觸發交易的預期報酬門檻值。",
    "trades": "總交易次數。",
    "total_return": "總報酬率 (幾何複利)。",
    "avg_return": "平均每筆交易的報酬率。",
    "win_rate": "勝率，報酬為正的交易筆數佔總交易筆數的比例。",
    "max_drawdown": "最大回撤，衡量策略可能出現的最大資金回落風險。",
    "annualized_return": "年化報酬率。",
    "sharpe": "夏普比率，衡量每單位風險所能帶來的超額報酬。",
    "short_trades": "做空交易的總次數。",
    "long_trades": "做多交易的總次數。",
    "short_total_return": "做空交易的總報酬率。",
    "long_total_return": "做多交易的總報酬率。",
    "short_win_rate": "做空交易的勝率。",
    "long_win_rate": "做多交易的勝率。",
    "mean_expected_short": "做空訊號的平均預期報酬。",
    "mean_expected_long": "做多訊號的平均預期報酬。",
    "short_avg_best_return": "做空交易在持有期內可能達到的平均最佳報酬。",
    "long_avg_best_return": "做多交易在持有期內可能達到的平均最佳報酬。",
    "score": "Optuna 優化時使用的目標分數 (此處為總報酬)。",
}

PERCENT_KEYS = {
    "accuracy",
    "mae_expected",
    "threshold",
    "total_return",
    "avg_return",
    "win_rate",
    "max_drawdown",
    "annualized_return",
    "short_total_return",
    "long_total_return",
    "short_win_rate",
    "long_win_rate",
    "mean_expected_short",
    "mean_expected_long",
    "short_avg_best_return",
    "long_avg_best_return",
    "score",
}


def main() -> None:
    from optimization.star_xgb_optuna import optimize_star_xgb

    args = _parse_args()

    result = optimize_star_xgb(
        symbol=args.symbol,
        timeframe=args.timeframe,
        lookback_days=args.lookback_days,
        test_days=args.test_days,
        exchange_id=args.exchange,
        model_dir=args.model_dir,
        params_store_path=args.params_store_path,
        trades_store_path=args.trades_store_path,
        n_trials=args.n_trials,
        timeout=args.timeout,
        study_name=args.study_name,
        storage=args.storage,
    )

    best_indicator_params = result.best_training_result.indicator_params.as_dict(
        rounded=True
    )
    best_model_params = result.best_training_result.model_params.as_dict(rounded=True)

    print("\n" + "=" * 80)
    print(" Optuna Optimization Finished".center(80))
    print("=" * 80)
    print(f"\nStudy: {result.study.study_name}")
    print(f"Best Value (Validation Total Return): {result.study.best_value:.4f}")
    print(
        "\n"
        + _format_params_as_table(best_indicator_params, "Best Indicator Parameters")
    )
    print(
        "\n" + _format_params_as_table(best_model_params, "Best Model Hyperparameters")
    )
    print(f"\nModel saved to: {result.best_training_result.model_path}")
    _render_section("Backtest Results: Train Set", result.train_backtest)
    _render_section("Backtest Results: Validation Set", result.valid_backtest)
    _render_section("Backtest Results: Test Set", result.test_backtest)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run star_xgb strategy optimization with Optuna"
    )
    parser.add_argument("--symbol", type=str, default="BTC/USD")
    parser.add_argument("--timeframe", type=str, default="5m")
    parser.add_argument("--lookback-days", type=int, default=360)
    parser.add_argument("--test-days", type=int, default=30)
    parser.add_argument("--exchange", type=str, default="binance")
    parser.add_argument(
        "--params-store-path", type=Path, default=Path("storage/strategy_state.db")
    )
    parser.add_argument(
        "--trades-store-path", type=Path, default=Path("storage/strategy_state.db")
    )
    parser.add_argument(
        "--model-dir", type=Path, default=Path("storage/models/star_xgb")
    )
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--study-name", type=str, default="star_xgb_default")
    parser.add_argument(
        "--storage", type=str, default="sqlite:///storage/optuna_studies.db"
    )
    return parser.parse_args()


def _format_params_as_table(params: Dict, title: str) -> str:
    header = f"--- {title} ---"
    keys = " | ".join(params.keys())
    values = " | ".join(map(str, params.values()))
    return f"{header}\n{keys}\n{values}"


def _format_metrics_and_explain(metrics: Dict) -> str:
    if not metrics:
        return "(No metrics to display)"

    formatted_metrics: list[str] = []
    seen: set[str] = set()
    for key, value in metrics.items():
        if key in seen or key in {"period_start", "period_end"}:
            continue
        seen.add(key)

        if isinstance(value, float):
            formatted_value = (
                f"{value * 100:.2f}%" if key in PERCENT_KEYS else f"{value:.4f}"
            )
        else:
            formatted_value = str(value)

        formatted_metrics.append(
            f"  - {key:<25}: {formatted_value} < {EXPLANATIONS.get(key, 'No description')}"
        )

    return "\n".join(formatted_metrics)


def _render_section(title: str, backtest_result) -> None:
    metrics = (
        backtest_result.metrics if backtest_result and backtest_result.metrics else {}
    )
    start = metrics.get("period_start")
    end = metrics.get("period_end")
    header = f"{title} [{start} ~ {end}]" if start and end else title

    print("\n" + "-" * 35)
    print(f" {header}".center(35))
    print("-" * 35)

    metrics_for_display = {
        k: v for k, v in metrics.items() if k not in {"period_start", "period_end"}
    }
    print(_format_metrics_and_explain(metrics_for_display))


if __name__ == "__main__":
    main()
