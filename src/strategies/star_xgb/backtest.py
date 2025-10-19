"""star_xgb 策略的端到端回測功能。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from .dataset import build_training_dataset, list_feature_columns
from .features import StarFeatureCache
from .labels import build_label_frame
from .model import _evaluate, CLASS_VALUES, _simulate_trades, _summarize_trades
from .params import StarIndicatorParams, StarModelParams
from .runtime import load_star_model


@dataclass
class StarBacktestResult:
    """儲存 star_xgb 回測結果的容器。"""

    trades: pd.DataFrame
    metrics: Dict[str, float]
    equity_curve: pd.DataFrame
    period_start: Optional[pd.Timestamp]
    period_end: Optional[pd.Timestamp]


def backtest_star_xgb(
    ohlcv: pd.DataFrame,
    indicator_params: StarIndicatorParams,
    model_params: StarModelParams,
    model_path: str,
    timeframe: str,
    class_means: List[float],
    class_thresholds: Dict[str, float],
    feature_columns: Optional[List[str]] = None,
    *,
    transaction_cost: float = 0.0,
    stop_loss_pct: Optional[float] = 0.005,
) -> StarBacktestResult:
    """
    執行 star_xgb 策略的端到端回測。
    """
    if ohlcv.empty:
        return StarBacktestResult(
            trades=pd.DataFrame(),
            metrics={},
            equity_curve=pd.DataFrame(),
            period_start=None,
            period_end=None,
        )

    cache = StarFeatureCache(
        ohlcv,
        trend_windows=[indicator_params.trend_window],
        atr_windows=[indicator_params.atr_window],
        volatility_windows=[indicator_params.volatility_window],
        volume_windows=[indicator_params.volume_window],
        pattern_windows=[indicator_params.pattern_lookback],
    )
    features = cache.build_features(indicator_params)
    label_thresholds = class_thresholds if class_thresholds else None
    labels, thresholds_used = build_label_frame(
        features, indicator_params, thresholds=label_thresholds
    )
    dataset = build_training_dataset(features, labels, class_thresholds=thresholds_used)

    if dataset.empty:
        return StarBacktestResult(
            trades=pd.DataFrame(),
            metrics={},
            equity_curve=pd.DataFrame(),
            period_start=None,
            period_end=None,
        )

    booster = load_star_model(model_path)

    if feature_columns is None:
        feature_columns = list_feature_columns(dataset)

    feature_array = dataset[feature_columns].to_numpy(dtype=float, copy=False)
    probs = booster.predict(feature_array)
    probs = probs.reshape(len(dataset), -1)

    class_means_arr = np.asarray(class_means, dtype=float)
    metrics = _evaluate(
        dataset,
        probs,
        model_params,
        class_means_arr,
        transaction_cost=transaction_cost,
        stop_loss_pct=stop_loss_pct,
    )

    trades = _build_signal_records(
        dataset,
        feature_columns,
        booster,
        model_params,
        indicator_params,
        class_means,
        timeframe,
        transaction_cost=transaction_cost,
        stop_loss_pct=stop_loss_pct,
    )

    equity_curve = _build_equity_curve(trades)

    trade_summary = _summarize_trades(trades)
    metrics.update(trade_summary)

    # 以資料期間計算年化報酬與夏普比率
    metrics["annualized_return"] = 0.0
    metrics["sharpe"] = 0.0
    if not trades.empty and not equity_curve.empty and not ohlcv.empty:
        timestamps = pd.to_datetime(ohlcv["timestamp"], utc=True, errors="coerce")
        if timestamps.notna().any():
            data_start = timestamps.min()
            data_end = timestamps.max()
        else:
            data_start = data_end = pd.NaT
        if pd.notna(data_start) and pd.notna(data_end) and data_end > data_start:
            period_days = (data_end - data_start).total_seconds() / 86400
            if period_days > 0:
                total_geo_return = trade_summary.get("total_return", 0.0)
                metrics["annualized_return"] = float(
                    (1 + total_geo_return) ** (365 / period_days) - 1
                )

                equity_series = equity_curve.set_index("timestamp")[
                    "equity"
                ].sort_index()
                baseline_index = data_start.normalize()
                if baseline_index.tzinfo is None and data_start.tzinfo is not None:
                    baseline_index = baseline_index.tz_localize(data_start.tzinfo)
                baseline = pd.Series([1.0], index=[baseline_index])
                equity_series = (
                    pd.concat([baseline, equity_series]).sort_index().ffill()
                )
                day_index = pd.date_range(
                    data_start.normalize(),
                    data_end.normalize(),
                    freq="D",
                    tz=data_start.tz,
                )
                if len(day_index) >= 2:
                    equity_daily = equity_series.reindex(day_index, method="ffill")
                    daily_returns = equity_daily.pct_change().dropna()
                    if not daily_returns.empty and daily_returns.std(ddof=0) > 0:
                        sharpe = (
                            daily_returns.mean()
                            / daily_returns.std(ddof=0)
                            * np.sqrt(365)
                        )
                        metrics["sharpe"] = float(sharpe)
    metrics["score"] = metrics.get("total_return", 0.0)

    timestamps = pd.to_datetime(ohlcv["timestamp"], utc=True, errors="coerce")
    if timestamps.notna().any():
        period_start = timestamps.min()
        period_end = timestamps.max()
    else:
        period_start = None
        period_end = None

    metrics["period_start"] = (
        period_start.isoformat() if isinstance(period_start, pd.Timestamp) else None
    )
    metrics["period_end"] = (
        period_end.isoformat() if isinstance(period_end, pd.Timestamp) else None
    )

    return StarBacktestResult(
        trades=trades,
        metrics=metrics,
        equity_curve=equity_curve,
        period_start=period_start,
        period_end=period_end,
    )


def _build_signal_records(
    dataset: pd.DataFrame,
    feature_columns: List[str],
    booster: lgb.Booster,
    model_params: StarModelParams,
    indicator_params: StarIndicatorParams,
    class_means: List[float],
    timeframe: str,
    *,
    transaction_cost: float = 0.0,
    stop_loss_pct: Optional[float] = None,
) -> pd.DataFrame:
    """從模型預測中建立交易記錄。"""
    if dataset.empty:
        return pd.DataFrame()

    features = dataset[feature_columns].to_numpy(dtype=float, copy=False)
    probs = booster.predict(features)
    probs = probs.reshape(len(dataset), -1)

    class_means_arr = np.asarray(class_means, dtype=float)
    expected_returns = probs @ class_means_arr
    pred_idx = probs.argmax(axis=1)
    pred_class = CLASS_VALUES[pred_idx]

    trades = _simulate_trades(
        dataset,
        expected_returns,
        pred_class,
        model_params,
        transaction_cost=transaction_cost,
        stop_loss_pct=stop_loss_pct,
    )
    return trades


def _build_equity_curve(trades: pd.DataFrame) -> pd.DataFrame:
    """根據交易記錄建立權益曲線。"""
    if trades.empty or "return" not in trades.columns:
        return pd.DataFrame(columns=["timestamp", "equity"])

    closed = trades.dropna(subset=["exit_time", "return"]).copy()
    if closed.empty:
        return pd.DataFrame(columns=["timestamp", "equity"])

    closed = closed.sort_values("exit_time")
    returns = pd.to_numeric(closed["return"], errors="coerce").fillna(0.0)
    equity_values = (1 + returns).cumprod()

    return pd.DataFrame({"timestamp": closed["exit_time"], "equity": equity_values})
