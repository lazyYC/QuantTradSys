"""star_xgb 策略特徵建構模組。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .params import StarIndicatorParams


@dataclass
class StarFeatureCache:
    """預先計算常用 rolling 指標，避免重複運算。"""

    base: pd.DataFrame
    trend_windows: Sequence[int]
    atr_windows: Sequence[int]
    volatility_windows: Sequence[int]
    volume_windows: Sequence[int]
    pattern_windows: Sequence[int]

    def __post_init__(self) -> None:
        _validate_required_columns(self.base)
        if self.base.empty:
            raise ValueError("Feature cache cannot be initialised with empty dataframe")

        frame = self.base.reset_index(drop=True).copy()
        self._base = frame
        closes = frame["close"].astype(float)
        highs = frame["high"].astype(float)
        lows = frame["low"].astype(float)
        volumes = frame["volume"].astype(float)

        self._trend_ma: Dict[int, pd.Series] = {}
        for window in sorted(set(int(w) for w in self.trend_windows)):
            if window <= 0:
                raise ValueError("trend window must be positive")
            self._trend_ma[window] = closes.rolling(
                window=window, min_periods=window
            ).mean()

        self._atr: Dict[int, pd.Series] = {}
        for window in sorted(set(int(w) for w in self.atr_windows)):
            if window <= 0:
                raise ValueError("ATR window must be positive")
            self._atr[window] = _compute_atr(frame, window)

        self._volatility_std: Dict[int, pd.Series] = {}
        for window in sorted(set(int(w) for w in self.volatility_windows)):
            if window <= 0:
                raise ValueError("volatility window must be positive")
            self._volatility_std[window] = closes.rolling(
                window=window, min_periods=window
            ).std(ddof=0)

        self._volume_mean: Dict[int, pd.Series] = {}
        for window in sorted(set(int(w) for w in self.volume_windows)):
            if window <= 0:
                raise ValueError("volume window must be positive")
            self._volume_mean[window] = volumes.rolling(
                window=window, min_periods=window
            ).mean()

        self._rolling_high: Dict[int, pd.Series] = {}
        self._rolling_low: Dict[int, pd.Series] = {}
        for window in sorted(set(int(w) for w in self.pattern_windows)):
            if window <= 0:
                raise ValueError("pattern window must be positive")
            self._rolling_high[window] = highs.rolling(
                window=window, min_periods=window
            ).max()
            self._rolling_low[window] = lows.rolling(
                window=window, min_periods=window
            ).min()

        identity = (
            len(frame),
            frame["timestamp"].iloc[0],
            frame["timestamp"].iloc[-1],
        )
        self._identity = identity

    def build_features(
        self,
        params: StarIndicatorParams,
        df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """依據參數組裝最終特徵表。"""
        base = self._base if df is None else df.reset_index(drop=True)
        if df is not None:
            _validate_required_columns(df)
            identity = (
                len(base),
                base["timestamp"].iloc[0],
                base["timestamp"].iloc[-1],
            )
            if identity != self._identity:
                raise ValueError("特徵快取與輸入資料不一致，請重新建立快取。")

        frame = base[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        open_ = frame["open"].astype(float)
        close = frame["close"].astype(float)
        high = frame["high"].astype(float)
        low = frame["low"].astype(float)
        volume = frame["volume"].astype(float)

        total_range = (high - low).replace(0.0, np.nan)
        upper_shadow = (high - np.maximum(close, open_)).clip(lower=0.0)
        lower_shadow = (np.minimum(close, open_) - low).clip(lower=0.0)
        body = (close - open_).abs()

        frame["upper_shadow_ratio"] = upper_shadow / total_range
        frame["lower_shadow_ratio"] = lower_shadow / total_range
        frame["body_ratio"] = body / total_range
        frame["body_direction"] = np.sign(close - open_)
        frame["intraday_range"] = (high - low) / close.replace(0.0, np.nan)

        trend_ma = self._trend_ma[params.trend_window]
        atr = self._atr[params.atr_window]
        volatility = self._volatility_std[params.volatility_window]
        volume_mean = self._volume_mean[params.volume_window]
        high_window = self._rolling_high[params.pattern_lookback]
        low_window = self._rolling_low[params.pattern_lookback]

        frame["trend_ma"] = trend_ma
        frame["trend_slope"] = (trend_ma - trend_ma.shift(params.slope_window)) / max(
            params.slope_window, 1
        )
        frame["close_trend_pct"] = (close - trend_ma) / trend_ma.replace(0.0, np.nan)
        frame["atr"] = atr
        frame["atr_norm"] = atr / close.replace(0.0, np.nan)
        frame["volatility_ratio"] = volatility / atr.replace(0.0, np.nan)
        frame["volume_ratio"] = volume / volume_mean.replace(0.0, np.nan)

        frame["rolling_high"] = high_window
        frame["rolling_low"] = low_window
        frame["proximity_to_high"] = (high_window - close) / high_window.replace(
            0.0, np.nan
        )
        frame["proximity_to_low"] = (close - low_window) / low_window.replace(
            0.0, np.nan
        )
        frame["range_percent"] = (high_window - low_window) / close.replace(0.0, np.nan)

        frame["return_1"] = close.pct_change()
        frame["return_3"] = close.pct_change(3)
        frame["return_6"] = close.pct_change(6)
        frame["volume_change_1"] = volume.pct_change()
        frame["shadow_diff"] = frame["upper_shadow_ratio"] - frame["lower_shadow_ratio"]

        frame = frame.dropna().reset_index(drop=True)
        return frame


def build_feature_frame(
    cache: StarFeatureCache,
    params: StarIndicatorParams,
    df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """方便函式：透過快取建構特徵 DataFrame。"""
    return cache.build_features(params, df=df)


def _compute_atr(df: pd.DataFrame, window: int) -> pd.Series:
    """計算 Average True Range。"""
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr_components = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1)
    return true_range.rolling(window=window, min_periods=window).mean()


def _validate_required_columns(df: pd.DataFrame) -> None:
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"缺少必要欄位: {sorted(missing)}")


__all__ = ["StarFeatureCache", "build_feature_frame"]
