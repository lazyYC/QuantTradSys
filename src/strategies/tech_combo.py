import itertools
import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from notifier.dispatcher import dispatch_signal
from signals.generator import FeatureStep, Scorer, generate_signal_frame

LOGGER = logging.getLogger(__name__)

PERIODS_PER_YEAR = int(365 * 24 * 60 / 5)  # 5 分鐘資料轉換為年化倍率


@dataclass(frozen=True)
class StrategyParams:
    ma_fast: int
    ma_slow: int
    rsi_period: int
    rsi_threshold: float


def _compute_rsi(series: pd.Series, period: int) -> pd.Series:
    """以平滑平均計算 RSI 指標。"""
    delta = series.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / period, adjust=False).mean()
    loss = -delta.clip(upper=0.0).ewm(alpha=1.0 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def make_feature_steps(params: StrategyParams) -> List[FeatureStep]:
    """依據參數產生特徵工程步驟。"""
    def add_fast_ma(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.assign(ma_fast=frame["close"].rolling(window=params.ma_fast).mean())

    def add_slow_ma(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.assign(ma_slow=frame["close"].rolling(window=params.ma_slow).mean())

    def add_rsi(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.assign(rsi=_compute_rsi(frame["close"], params.rsi_period))

    return [add_fast_ma, add_slow_ma, add_rsi]


def make_scorer(params: StrategyParams) -> Scorer:
    """產生依據 MA 與 RSI 判斷的評分器。"""
    def scorer(row: pd.Series):
        if pd.isna(row.ma_fast) or pd.isna(row.ma_slow) or pd.isna(row.rsi):
            return None
        if row.ma_fast > row.ma_slow and row.rsi >= params.rsi_threshold:
            return {"name": "ma_rsi_combo", "decision": "BUY", "confidence": 1.0}
        if row.ma_fast < row.ma_slow and row.rsi <= (100 - params.rsi_threshold):
            return {"name": "ma_rsi_combo", "decision": "SELL", "confidence": 1.0}
        return None

    return scorer


def decisions_to_position(decisions: Sequence[str]) -> pd.Series:
    """根據訊號序列回推部位變化。"""
    position = 0
    position_trace: List[int] = []
    for decision in decisions:
        if decision == "BUY":
            position = 1
        elif decision == "SELL":
            position = 0
        position_trace.append(position)
    return pd.Series(position_trace)


def evaluate_strategy(df: pd.DataFrame, params: StrategyParams) -> Dict[str, float]:
    """回測指定參數組合並計算績效指標。"""
    feature_steps = make_feature_steps(params)
    scorer = make_scorer(params)
    signal_df = generate_signal_frame(df, feature_steps, [scorer])
    signal_df = signal_df.reset_index(drop=True)

    price_returns = signal_df["close"].pct_change().fillna(0.0)
    positions = decisions_to_position(signal_df["signal"]).shift(1).fillna(0)
    strategy_returns = price_returns * positions

    cumulative = (1 + strategy_returns).cumprod()
    total_return = cumulative.iloc[-1] - 1 if not cumulative.empty else 0.0
    periods = max(len(strategy_returns), 1)
    annualized = (1 + total_return) ** (PERIODS_PER_YEAR / periods) - 1
    volatility = strategy_returns.std(ddof=0)
    sharpe = (strategy_returns.mean() / volatility * np.sqrt(PERIODS_PER_YEAR)) if volatility > 0 else 0.0

    running_max = cumulative.cummax()
    drawdown = (cumulative / running_max - 1).min() if not cumulative.empty else 0.0

    trade_count = int((signal_df["signal"] == "BUY").sum())

    return {
        "annualized_return": float(annualized),
        "total_return": float(total_return),
        "sharpe": float(sharpe),
        "max_drawdown": float(abs(drawdown)),
        "trades": trade_count,
    }


def grid_search(
    df: pd.DataFrame,
    ma_fast_range: Iterable[int],
    ma_slow_range: Iterable[int],
    rsi_period_range: Iterable[int],
    rsi_threshold_range: Iterable[float],
    top_n: int = 5,
) -> List[Dict[str, float]]:
    """遍歷指標參數，回傳績效最佳的組合。"""
    results: List[Dict[str, float]] = []
    for ma_fast, ma_slow, rsi_period, rsi_th in itertools.product(
        ma_fast_range, ma_slow_range, rsi_period_range, rsi_threshold_range
    ):
        if ma_fast >= ma_slow:
            continue
        params = StrategyParams(ma_fast=ma_fast, ma_slow=ma_slow, rsi_period=rsi_period, rsi_threshold=rsi_th)
        metrics = evaluate_strategy(df, params)
        metrics.update({
            "ma_fast": ma_fast,
            "ma_slow": ma_slow,
            "rsi_period": rsi_period,
            "rsi_threshold": rsi_th,
        })
        results.append(metrics)
    results.sort(key=lambda item: item["annualized_return"], reverse=True)
    return results[:top_n]


def generate_realtime_signal(df: pd.DataFrame, params: StrategyParams) -> str:
    """以最佳參數產出最新訊號並透過通知元件發佈。"""
    feature_steps = make_feature_steps(params)
    scorer = make_scorer(params)
    signal_subset = df.tail(max(params.ma_slow, params.rsi_period) + 5).reset_index(drop=True)
    signal_df = generate_signal_frame(signal_subset, feature_steps, [scorer])
    latest_row = signal_df.iloc[-1]
    decision = latest_row["signal"]
    dispatch_signal(decision, {
        "timestamp": str(latest_row["timestamp"]),
        "close": float(latest_row["close"]),
        "params": params.__dict__,
    })
    return decision


__all__ = [
    "StrategyParams",
    "evaluate_strategy",
    "grid_search",
    "generate_realtime_signal",
    "make_feature_steps",
    "make_scorer",
]

