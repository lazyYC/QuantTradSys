import logging
from dataclasses import dataclass, asdict
from typing import (
    Any,
    ClassVar,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import pandas as pd

from utils.formatting import format_params, round_numeric_fields

LOGGER = logging.getLogger(__name__)

PERIODS_PER_YEAR = int(365 * 24 * 60 / 5)

CONTEXT_DECIMALS = {
    "close": 2,
    "atr": 3,
    "zscore": 3,
    "entry_price": 2,
    "entry_zscore": 3,
    "pnl_pct": 6,
}


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

    _ROUND_DECIMALS: ClassVar[Dict[str, int]] = {
        "bb_std": 1,
        "atr_mult": 1,
        "entry_zscore": 1,
        "volume_z": 1,
        "stop_loss_mult": 1,
        "exit_zscore": 1,
    }

    def as_dict(self, rounded: bool = False) -> dict:
        data = asdict(self)
        if not rounded:
            return data
        return format_params(data, decimals_map=self._ROUND_DECIMALS)

    def __repr__(self) -> str:
        ordered_keys = (
            "sma_window",
            "bb_std",
            "atr_window",
            "atr_mult",
            "entry_zscore",
            "volume_window",
            "volume_z",
            "pattern_min",
            "stop_loss_mult",
            "exit_zscore",
        )
        rounded = self.as_dict(rounded=True)
        body = ", ".join(f"{key}={rounded[key]}" for key in ordered_keys)
        return f"MeanReversionParams({body})"


@dataclass(frozen=True)
class MeanReversionRuntimeState:
    position: int = 0
    entry_price: float = 0.0
    entry_time: Optional[pd.Timestamp] = None
    entry_zscore: float = 0.0

    def to_dict(self) -> dict:
        return {
            "position": self.position,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat()
            if self.entry_time is not None
            else None,
            "entry_zscore": self.entry_zscore,
        }

    @classmethod
    def from_dict(
        cls, data: Optional[Mapping[str, Any]]
    ) -> "MeanReversionRuntimeState":
        if not data:
            return cls()
        entry_time = data.get("entry_time")
        timestamp = pd.to_datetime(entry_time) if entry_time else None
        if isinstance(timestamp, pd.Timestamp):
            if pd.isna(timestamp):
                timestamp = None
            elif timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize("UTC")
        return cls(
            position=int(data.get("position", 0)),
            entry_price=float(data.get("entry_price", 0.0)),
            entry_time=timestamp,
            entry_zscore=float(data.get("entry_zscore", 0.0)),
        )


@dataclass(frozen=True)
class BacktestResult:
    metrics: Dict[str, float]
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    per_bar_returns: pd.Series


class MeanReversionFeatureCache:
    """預先計算技術指標，避免大量試驗時重複運算。"""

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        sma_windows: Sequence[int],
        atr_windows: Sequence[int],
        volume_windows: Sequence[int],
        pattern_windows: Sequence[int],
    ) -> None:
        if df.empty:
            raise ValueError("Cannot initialise feature cache with empty dataframe")
        self._base = df.reset_index(drop=True).copy()
        self._identity = (
            len(self._base),
            self._base["timestamp"].iloc[0],
            self._base["timestamp"].iloc[-1],
        )
        closes = self._base["close"]
        opens = self._base["open"]
        volumes = self._base.get("volume", pd.Series(np.zeros(len(self._base))))

        self._sma: Dict[int, pd.Series] = {}
        self._std: Dict[int, pd.Series] = {}
        for window in sorted(set(sma_windows)):
            self._sma[window] = closes.rolling(window=window, min_periods=window).mean()
            self._std[window] = closes.rolling(window=window, min_periods=window).std(
                ddof=0
            )

        self._atr: Dict[int, pd.Series] = {}
        for window in sorted(set(atr_windows)):
            self._atr[window] = compute_atr(self._base, window)

        self._volume_mean: Dict[int, pd.Series] = {}
        self._volume_std: Dict[int, pd.Series] = {}
        for window in sorted(set(volume_windows)):
            self._volume_mean[window] = volumes.rolling(
                window=window, min_periods=window
            ).mean()
            self._volume_std[window] = volumes.rolling(
                window=window, min_periods=window
            ).std(ddof=0)

        bearish = (closes < opens).astype(int)
        bullish = (closes > opens).astype(int)
        self._bear_counts: Dict[int, pd.Series] = {}
        self._bull_counts: Dict[int, pd.Series] = {}
        for window in sorted(set(pattern_windows)):
            self._bear_counts[window] = bearish.rolling(
                window=window, min_periods=window
            ).sum()
            self._bull_counts[window] = bullish.rolling(
                window=window, min_periods=window
            ).sum()

    def _validate_identity(self, df: pd.DataFrame) -> None:
        identity = (len(df), df["timestamp"].iloc[0], df["timestamp"].iloc[-1])
        if identity != self._identity:
            raise ValueError(
                "Feature cache dataframe mismatch; ensure identical slice is used"
            )

    def build_features(
        self, params: MeanReversionParams, df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        base = self._base if df is None else df
        if df is not None:
            self._validate_identity(df)
        frame = base[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        frame["sma"] = self._sma[params.sma_window]
        std = self._std[params.sma_window]
        frame["std"] = std
        frame["upper"] = frame["sma"] + params.bb_std * std
        frame["lower"] = frame["sma"] - params.bb_std * std
        frame["atr"] = self._atr[params.atr_window]
        frame["zscore"] = (frame["close"] - frame["sma"]) / std.replace(0, np.nan)
        vol_mean = self._volume_mean[params.volume_window]
        vol_std = self._volume_std[params.volume_window].replace(0, np.nan)
        frame["volume_z"] = (frame["volume"] - vol_mean) / vol_std
        frame["bear_count"] = self._bear_counts[params.pattern_min]
        frame["bull_count"] = self._bull_counts[params.pattern_min]
        frame = frame.dropna().reset_index(drop=True)
        return frame


def compute_features(df: pd.DataFrame, params: MeanReversionParams) -> pd.DataFrame:
    """建立均值回歸策略所需指標。"""
    frame = df.copy()
    frame["sma"] = (
        frame["close"]
        .rolling(window=params.sma_window, min_periods=params.sma_window)
        .mean()
    )
    frame["std"] = (
        frame["close"]
        .rolling(window=params.sma_window, min_periods=params.sma_window)
        .std(ddof=0)
    )
    frame["upper"] = frame["sma"] + params.bb_std * frame["std"]
    frame["lower"] = frame["sma"] - params.bb_std * frame["std"]
    frame["atr"] = compute_atr(frame, params.atr_window)
    frame["zscore"] = (frame["close"] - frame["sma"]) / frame["std"].replace(0, np.nan)
    volume_mean = (
        frame["volume"]
        .rolling(window=params.volume_window, min_periods=params.volume_window)
        .mean()
    )
    volume_std = (
        frame["volume"]
        .rolling(window=params.volume_window, min_periods=params.volume_window)
        .std(ddof=0)
    )
    frame["volume_z"] = (frame["volume"] - volume_mean) / volume_std.replace(0, np.nan)
    window = max(params.pattern_min, 1)
    frame["bear_count"] = (
        (frame["close"] < frame["open"])
        .rolling(window=window, min_periods=window)
        .sum()
    )
    frame["bull_count"] = (
        (frame["close"] > frame["open"])
        .rolling(window=window, min_periods=window)
        .sum()
    )
    frame = frame.dropna().reset_index(drop=True)
    return frame


def backtest_mean_reversion(
    df: pd.DataFrame,
    params: MeanReversionParams,
    feature_cache: Optional[MeanReversionFeatureCache] = None,
) -> BacktestResult:
    """執行單組參數的回測，返回績效與交易紀錄。"""
    features = (
        feature_cache.build_features(params, df)
        if feature_cache
        else compute_features(df, params)
    )
    if features.empty:
        empty_metrics = {
            "annualized_return": 0.0,
            "total_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "trades": 0,
        }
        return BacktestResult(
            empty_metrics, pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float)
        )

    position = 0
    entry_price = 0.0
    entry_time: Optional[pd.Timestamp] = None
    entry_z = 0.0
    trades: List[Dict[str, object]] = []
    equity = 1.0
    equity_curve: List[float] = []
    equity_timestamps: List[pd.Timestamp] = []
    per_bar_returns: List[float] = []
    prev_close: Optional[float] = None

    for _, row in features.iterrows():
        timestamp = row["timestamp"]
        close = row["close"]
        if prev_close is None:
            bar_ret = 0.0
        else:
            bar_ret = position * (close - prev_close) / prev_close
            equity *= 1 + bar_ret
        per_bar_returns.append(bar_ret)
        equity_curve.append(equity)
        equity_timestamps.append(timestamp)

        exit_flag, exit_reason = _exit_conditions(row, params, position, entry_price)
        if exit_flag and position != 0 and entry_time is not None:
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
                    "exit_zscore": row["zscore"],
                    "exit_reason": exit_reason,
                }
            )
            position = 0
            entry_time = None
            entry_price = 0.0

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
    per_bar_series = pd.Series(
        per_bar_returns, index=features["timestamp"], name="strategy_return"
    )
    if per_bar_series.empty:
        empty_metrics = {
            "annualized_return": 0.0,
            "total_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "trades": 0,
        }
        equity_df = pd.DataFrame(
            {"timestamp": features["timestamp"], "equity": equity_curve}
        )
        return BacktestResult(empty_metrics, trades_df, equity_df, per_bar_series)

    total_return = equity_curve[-1] - 1
    mean_ret = per_bar_series.mean()
    std_ret = per_bar_series.std(ddof=0)
    sharpe = (mean_ret / std_ret * np.sqrt(PERIODS_PER_YEAR)) if std_ret > 0 else 0.0

    equity_series = pd.Series(equity_curve, index=equity_timestamps)
    running_max = equity_series.cummax()
    drawdown = (
        ((equity_series / running_max) - 1).min() if not equity_series.empty else 0.0
    )

    win_rate = float((trades_df["return"] > 0).mean()) if not trades_df.empty else 0.0
    metrics = {
        "annualized_return": float(
            (1 + total_return) ** (PERIODS_PER_YEAR / max(len(per_bar_series), 1)) - 1
        ),
        "total_return": float(total_return),
        "sharpe": float(sharpe),
        "max_drawdown": float(abs(drawdown)),
        "win_rate": win_rate,
        "trades": int(len(trades_df)),
    }

    equity_df = equity_series.reset_index()
    equity_df.columns = ["timestamp", "equity"]

    return BacktestResult(metrics, trades_df, equity_df, per_bar_series)


def generate_realtime_decision(
    df: pd.DataFrame,
    params: MeanReversionParams,
    *,
    state: Optional[MeanReversionRuntimeState] = None,
) -> tuple[str, dict, MeanReversionRuntimeState]:
    """根據最新 K 線判斷即時訊號並回傳狀態。"""
    runtime_state = state or MeanReversionRuntimeState()
    features = compute_features(df, params)
    if features.empty:
        context = {
            "reason": "insufficient_data",
            "position_before": runtime_state.position,
            "position_after": runtime_state.position,
            "params": params.as_dict(rounded=True),
        }
        context = round_numeric_fields(context, decimals_map=CONTEXT_DECIMALS)
        return "HOLD", context, runtime_state

    latest = features.iloc[-1]
    timestamp = latest["timestamp"]
    close_price = float(latest["close"])
    zscore = float(latest["zscore"])
    atr_value = float(latest["atr"])

    context = {
        "timestamp": str(timestamp),
        "close": close_price,
        "zscore": zscore,
        "atr": atr_value,
        "position_before": runtime_state.position,
        "params": params.as_dict(rounded=True),
    }
    action = "HOLD"
    new_state = runtime_state

    if runtime_state.position != 0:
        exit_flag, exit_reason = _exit_conditions(
            latest, params, runtime_state.position, runtime_state.entry_price
        )
        if exit_flag:
            pnl_pct = 0.0
            if runtime_state.entry_price:
                pnl_pct = (
                    close_price - runtime_state.entry_price
                ) / runtime_state.entry_price
                pnl_pct *= runtime_state.position
            context.update(
                {
                    "reason": exit_reason,
                    "entry_price": runtime_state.entry_price,
                    "entry_time": runtime_state.entry_time.isoformat()
                    if runtime_state.entry_time is not None
                    else None,
                    "pnl_pct": pnl_pct,
                }
            )
            new_state = MeanReversionRuntimeState()
            action = "EXIT_LONG" if runtime_state.position == 1 else "EXIT_SHORT"

    if action == "HOLD" and runtime_state.position == 0:
        long_cond, short_cond = _entry_conditions(latest, params)
        if long_cond:
            action = "ENTER_LONG"
            new_state = MeanReversionRuntimeState(
                position=1,
                entry_price=close_price,
                entry_time=timestamp,
                entry_zscore=zscore,
            )
            context["entry_zscore"] = zscore
        elif short_cond:
            action = "ENTER_SHORT"
            new_state = MeanReversionRuntimeState(
                position=-1,
                entry_price=close_price,
                entry_time=timestamp,
                entry_zscore=zscore,
            )
            context["entry_zscore"] = zscore

    context["position_after"] = new_state.position
    context = round_numeric_fields(context, decimals_map=CONTEXT_DECIMALS)
    return action, context, new_state


def grid_search_mean_reversion(
    df: pd.DataFrame,
    param_grid: Iterable[MeanReversionParams],
    feature_cache: Optional[MeanReversionFeatureCache] = None,
) -> List[Dict[str, object]]:
    """遍歷參數組合並以年化報酬排序。"""
    results: List[Dict[str, object]] = []
    for params in param_grid:
        result = backtest_mean_reversion(df, params, feature_cache=feature_cache)
        metrics = result.metrics
        metrics.update({"params": params})
        results.append(metrics)
    results.sort(key=lambda item: item.get("annualized_return", 0.0), reverse=True)
    return results


__all__ = [
    "MeanReversionParams",
    "MeanReversionRuntimeState",
    "BacktestResult",
    "MeanReversionFeatureCache",
    "compute_features",
    "backtest_mean_reversion",
    "generate_realtime_decision",
    "grid_search_mean_reversion",
]


def compute_atr(df: pd.DataFrame, window: int) -> pd.Series:
    """計算 Average True Range。"""
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=window).mean()


def _entry_conditions(row: pd.Series, params: MeanReversionParams) -> Tuple[bool, bool]:
    long_cond = (row["zscore"] <= -params.entry_zscore) or (
        (row["sma"] - row["close"]) >= params.atr_mult * row["atr"]
    )
    long_cond = (
        long_cond
        and (row["volume_z"] >= params.volume_z)
        and (row["bear_count"] >= params.pattern_min)
    )
    short_cond = (row["zscore"] >= params.entry_zscore) or (
        (row["close"] - row["sma"]) >= params.atr_mult * row["atr"]
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
