import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MeanReversionParams:
    sma_window: int
    bb_std: float
    atr_window: int
    atr_mult: float
    entry_zscore: float
    volume_window: int
    volume_z: float
    pattern_min: int
    stop_loss_mult: float
    exit_zscore: float


@dataclass(frozen=True)
class BacktestResult:
    metrics: Dict[str, float]
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    per_bar_returns: pd.Series


def compute_atr(df: pd.DataFrame, window: int) -> pd.Series:
    """計算 Average True Range。"""
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=window).mean()


def compute_features(df: pd.DataFrame, params: MeanReversionParams) -> pd.DataFrame:
    """建立均值回歸策略所需指標。"""
    frame = df.copy()
    frame["sma"] = frame["close"].rolling(window=params.sma_window, min_periods=params.sma_window).mean()
    frame["std"] = frame["close"].rolling(window=params.sma_window, min_periods=params.sma_window).std(ddof=0)
    frame["upper"] = frame["sma"] + params.bb_std * frame["std"]
    frame["lower"] = frame["sma"] - params.bb_std * frame["std"]
    frame["atr"] = compute_atr(frame, params.atr_window)
    frame["zscore"] = (frame["close"] - frame["sma"]) / frame["std"].replace(0, np.nan)
    volume_mean = frame["volume"].rolling(window=params.volume_window, min_periods=params.volume_window).mean()
    volume_std = frame["volume"].rolling(window=params.volume_window, min_periods=params.volume_window).std(ddof=0)
    frame["volume_z"] = (frame["volume"] - volume_mean) / volume_std.replace(0, np.nan)
    frame["bear_count"] = (frame["close"] < frame["open"]).rolling(window=3, min_periods=3).sum()
    frame["bull_count"] = (frame["close"] > frame["open"]).rolling(window=3, min_periods=3).sum()
    frame = frame.dropna().reset_index(drop=True)
    return frame


def _entry_conditions(row: pd.Series, params: MeanReversionParams) -> Tuple[bool, bool]:
    long_cond = (
        (row["zscore"] <= -params.entry_zscore)
        or ((row["sma"] - row["close"]) >= params.atr_mult * row["atr"])
    )
    long_cond = (
        long_cond
        and (row["volume_z"] >= params.volume_z)
        and (row["bear_count"] >= params.pattern_min)
    )
    short_cond = (
        (row["zscore"] >= params.entry_zscore)
        or ((row["close"] - row["sma"]) >= params.atr_mult * row["atr"])
    )
    short_cond = (
        short_cond
        and (row["volume_z"] >= params.volume_z)
        and (row["bull_count"] >= params.pattern_min)
    )
    return long_cond, short_cond


def _exit_conditions(
    row: pd.Series,
    params: MeanReversionParams,
    position: int,
    entry_price: float,
) -> Tuple[bool, str]:
    if position == 0:
        return False, ""
    z = row["zscore"]
    atr = row["atr"]
    price = row["close"]
    if position == 1:
        if z >= -params.exit_zscore:
            return True, "mean_revert"
        if price <= entry_price - params.stop_loss_mult * atr:
            return True, "stop_loss"
    else:
        if z <= params.exit_zscore:
            return True, "mean_revert"
        if price >= entry_price + params.stop_loss_mult * atr:
            return True, "stop_loss"
    return False, ""


def backtest_mean_reversion(df: pd.DataFrame, params: MeanReversionParams) -> BacktestResult:
    """執行單組參數的回測，返回績效與交易紀錄。"""
    features = compute_features(df, params)
    if features.empty:
        empty_metrics = {
            "annualized_return": 0.0,
            "total_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "trades": 0,
        }
        return BacktestResult(empty_metrics, pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float))

    position = 0
    entry_price = 0.0
    entry_time: pd.Timestamp | None = None
    entry_z = 0.0
    trades: List[Dict[str, object]] = []
    equity = 1.0
    equity_curve: List[float] = []
    equity_timestamps: List[pd.Timestamp] = []
    per_bar_returns: List[float] = []
    prev_close = None

    for idx, row in features.iterrows():
        timestamp = row["timestamp"]
        close = row["close"]
        if prev_close is None:
            bar_ret = 0.0
        else:
            bar_ret = position * (close - prev_close) / prev_close
            equity *= (1 + bar_ret)
        per_bar_returns.append(bar_ret)
        equity_curve.append(equity)
        equity_timestamps.append(timestamp)

        # 先檢查平倉條件
        exit_flag, exit_reason = _exit_conditions(row, params, position, entry_price)
        if exit_flag and position != 0:
            pnl = (close - entry_price) / entry_price * position
            holding = (timestamp - entry_time).total_seconds() / 60 if entry_time else 0.0
            trades.append(
                {
                    "entry_time": entry_time,
                    "exit_time": timestamp,
                    "side": "LONG" if position == 1 else "SHORT",
                    "entry_price": entry_price,
                    "exit_price": close,
                    "return": pnl,
                    "holding_mins": holding,
                    "entry_zscore": entry_z,
                    "exit_zscore": row["zscore"],
                    "exit_reason": exit_reason,
                }
            )
            position = 0
            entry_time = None
            entry_price = 0.0

        # 再檢查開倉條件
        if position == 0:
            long_cond, short_cond = _entry_conditions(row, params)
            if long_cond:
                position = 1
                entry_price = close
                entry_time = timestamp
                entry_z = row["zscore"]
            elif short_cond:
                position = -1
                entry_price = close
                entry_time = timestamp
                entry_z = row["zscore"]

        prev_close = close

    # 期末強制平倉
    if position != 0 and entry_time is not None:
        close = features["close"].iloc[-1]
        timestamp = features["timestamp"].iloc[-1]
        pnl = (close - entry_price) / entry_price * position
        holding = (timestamp - entry_time).total_seconds() / 60
        trades.append(
            {
                "entry_time": entry_time,
                "exit_time": timestamp,
                "side": "LONG" if position == 1 else "SHORT",
                "entry_price": entry_price,
                "exit_price": close,
                "return": pnl,
                "holding_mins": holding,
                "entry_zscore": entry_z,
                "exit_zscore": features["zscore"].iloc[-1],
                "exit_reason": "forced_exit",
            }
        )

    trades_df = pd.DataFrame(trades)
    per_bar_series = pd.Series(per_bar_returns, index=features["timestamp"], name="strategy_return")
    if per_bar_series.empty:
        empty_metrics = {
            "annualized_return": 0.0,
            "total_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "trades": 0,
        }
        equity_df = pd.DataFrame({"timestamp": features["timestamp"], "equity": equity_curve})
        return BacktestResult(empty_metrics, trades_df, equity_df, per_bar_series)

    total_return = equity_curve[-1] - 1
    periods_per_year = 365 * 24 * 60 / 5
    mean_ret = per_bar_series.mean()
    std_ret = per_bar_series.std(ddof=0)
    sharpe = (mean_ret / std_ret * np.sqrt(periods_per_year)) if std_ret > 0 else 0.0

    equity_series = pd.Series(equity_curve, index=equity_timestamps)
    running_max = equity_series.cummax()
    drawdown = ((equity_series / running_max) - 1).min() if not equity_series.empty else 0.0

    win_rate = float((trades_df["return"] > 0).mean()) if not trades_df.empty else 0.0
    metrics = {
        "annualized_return": float((1 + total_return) ** (periods_per_year / max(len(per_bar_series), 1)) - 1),
        "total_return": float(total_return),
        "sharpe": float(sharpe),
        "max_drawdown": float(abs(drawdown)),
        "win_rate": win_rate,
        "trades": int(len(trades_df)),
    }

    equity_df = equity_series.reset_index()
    equity_df.columns = ["timestamp", "equity"]

    return BacktestResult(metrics, trades_df, equity_df, per_bar_series)


def grid_search_mean_reversion(
    df: pd.DataFrame,
    param_grid: Iterable[MeanReversionParams],
) -> List[Dict[str, object]]:
    """遍歷參數組合並以年化報酬排序。"""
    results: List[Dict[str, object]] = []
    for params in param_grid:
        result = backtest_mean_reversion(df, params)
        metrics = result.metrics
        metrics.update({
            "params": params,
        })
        results.append(metrics)
    results.sort(key=lambda item: item.get("annualized_return", 0.0), reverse=True)
    return results


__all__ = [
    "MeanReversionParams",
    "BacktestResult",
    "compute_features",
    "backtest_mean_reversion",
    "grid_search_mean_reversion",
]
