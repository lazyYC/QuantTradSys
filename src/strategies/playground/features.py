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
        prev_close = close.shift(1)
        safe_prev_close = prev_close.replace(0.0, np.nan)

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
        trend_slope = (trend_ma - trend_ma.shift(params.slope_window)) / max(
            params.slope_window, 1
        )
        frame["trend_slope"] = trend_slope / trend_ma.replace(0.0, np.nan)
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
        
        # --- NEW FEATURES START ---
        
        # 1. RSI
        # Use cached if available (not implemented yet in cache input, so calc on fly for now or modify cache logic)
        # To strictly follow plan, we should have cached it. 
        # But for now, let's calc on fly for simplicity in this edit, as params are fixed.
        # Future optimization: Add to cache.
        rsi_s = _compute_rsi(close, params.rsi_window)
        frame["rsi"] = rsi_s
        frame["rsi_slope"] = (rsi_s - rsi_s.shift(3)) / 3.0 # Short term momentum of momentum
        
        # 2. Bollinger Bands
        # _compute_bb returns (upper, middle, lower)
        bb_up, bb_mid, bb_low = _compute_bb(close, params.bb_window, params.bb_std)
        bb_width = (bb_up - bb_low) / bb_mid.replace(0.0, np.nan)
        bb_pos = (close - bb_low) / (bb_up - bb_low).replace(0.0, np.nan)
        frame["bb_width"] = bb_width
        frame["bb_pos"] = bb_pos
        
        # 3. MACD
        macd_line, macd_signal, macd_hist = _compute_macd(close, params.macd_fast, params.macd_slow, params.macd_signal)
        # Normalize by close to make it price-agnostic
        frame["macd_hist_norm"] = macd_hist / close.replace(0.0, np.nan)
        
        # 4. Log Returns
        # safe log
        log_ret = np.log(close / close.shift(1).replace(0.0, np.nan))
        frame["log_return_1"] = log_ret
        frame["log_return_5"] = log_ret.rolling(5).sum()
        
        # 5. Trend Z-Score (Dynamic Deviation)
        # Calculate std for the trend window specifically to get true Z-Score
        trend_std = close.rolling(window=params.trend_window, min_periods=params.trend_window).std(ddof=0)
        frame["trend_z_score"] = (close - trend_ma) / trend_std.replace(0.0, np.nan)
        
        # 6. Short-term Deviations (MA5, MA15)
        # Capture short term overextension
        ma_5 = close.rolling(window=5, min_periods=5).mean()
        ma_15 = close.rolling(window=15, min_periods=15).mean()
        frame["dist_ma_5"] = (close - ma_5) / ma_5.replace(0.0, np.nan)
        frame["dist_ma_15"] = (close - ma_15) / ma_15.replace(0.0, np.nan)
        
        # 7. BB Width Delta
        # Expansion/Contraction
        frame["bb_width_delta"] = (bb_width - bb_width.shift(1)) / bb_width.shift(1).replace(0.0, np.nan)
        
        # 8. Support/Resistance Features
        # pos_in_range_{k}: Position within the high-low range of last k bars
        # range_dist_{k}: Range width relative to close (Volatility/Compression)
        # touch_count_{k}: How many times price touched near the extremes
        
        sr_windows = [12, 36, 72, 144, 288] # 1H, 3H, 6H, 12H, 24H
        
        for k in sr_windows:
            roll_high = high.rolling(window=k, min_periods=k).max()
            roll_low = low.rolling(window=k, min_periods=k).min()
            
            # 8.1 Position in Range (0.0=Low, 1.0=High)
            frame[f"pos_in_range_{k}"] = (close - roll_low) / (roll_high - roll_low).replace(0.0, np.nan)
            
            # 8.2 Range Compression/Expansion (Width %)
            # High value = Expanded, Low value = Compressed
            frame[f"range_dist_{k}"] = (roll_high - roll_low) / close.replace(0.0, np.nan)
        
        # 8.3 Touch Count (Test of Support/Resistance)
        # Count how many lows are within 0.2% of the rolling min (Support Tests)
        # Count how many highs are within 0.2% of the rolling max (Resistance Tests)
        # For touch_count, we usually look at a meaningful window like 72 (6H) or 288 (24H)
        # Doing it for 72 only to save compute
        touch_window = 72
        threshold_pct = 0.002
        
        roll_min_72 = low.rolling(window=touch_window, min_periods=touch_window).min()
        roll_max_72 = high.rolling(window=touch_window, min_periods=touch_window).max()
        
        # Boolean series: Is this bar's low near the 72-period low?
        # Note: We need to compare current low vs existing roll_min (which includes current).
        # A touch is defined as: low <= roll_min * (1 + thresh)
        low_touch = (low <= roll_min_72 * (1 + threshold_pct)).astype(float)
        high_touch = (high >= roll_max_72 * (1 - threshold_pct)).astype(float)
        
        # Sum touches in the window
        frame[f"support_touch_count_{touch_window}"] = low_touch.rolling(window=touch_window).sum()
        frame[f"resistance_touch_count_{touch_window}"] = high_touch.rolling(window=touch_window).sum()

        # 8.4 ADX (Trend Strength)
        # Using 14 period usually. reusing rsi_window or atr_window? 
        # Let's use 14 as standard.
        frame["adx"] = _compute_adx(high, low, close, 14)
        
        # --- NEW FEATURES END ---

        # Removed trade_amount (non-stationary)
        # frame["trade_amount"] = close * volume
        
        frame["open_rel"] = (open_ - safe_prev_close) / safe_prev_close
        frame["high_rel"] = (high - safe_prev_close) / safe_prev_close
        frame["low_rel"] = (low - safe_prev_close) / safe_prev_close
        frame["close_rel"] = (close - safe_prev_close) / safe_prev_close
        for lag in range(1, 6):
            frame[f"close_rel_lag{lag}"] = frame["close_rel"].shift(lag)

        frame["return_1"] = close.pct_change()
        frame["return_3"] = close.pct_change(3)
        frame["return_6"] = close.pct_change(6)
        vol_change = volume.pct_change()
        vol_change = vol_change.replace([np.inf, -np.inf], np.nan)
        frame["volume_change_1"] = np.sign(vol_change) * np.log1p(vol_change.abs())
        frame["shadow_diff"] = frame["upper_shadow_ratio"] - frame["lower_shadow_ratio"]

        frame = frame.drop(
            columns=[
                "rolling_high", "rolling_low", "atr", "volume", 
                "log_return_1", "log_return_5"
            ],
            errors="ignore",
        )
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


def _compute_rsi(series: pd.Series, window: int) -> pd.Series:
    """Compute RSI using Wilder's Smoothing."""
    diff = series.diff()
    gain = diff.clip(lower=0).replace(0, np.nan) # replace 0 with nan to init first average
    loss = -diff.clip(upper=0).replace(0, np.nan)
    
    # Standard Wilder's RSI calculation
    # First Average Gain
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    
    # For subsequent steps, use exponential smoothing (alpha=1/window)
    # Actually, Pandas ewm(com=window-1) is similar to Wilder
    # Wilder's smoothing alpha = 1/N. Pandas ewm alpha=1/(1+com). So com=N-1.
    avg_gain = gain.ewm(com=window-1, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(com=window-1, min_periods=window, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _compute_bb(series: pd.Series, window: int, std_dev: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Compute Bollinger Bands (Upper, Middle, Lower)."""
    middle = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std(ddof=0)
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower


def _compute_macd(series: pd.Series, fast: int, slow: int, signal: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Compute MACD, Signal, Hist."""
    # Fast / Slow EMA
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Compute ADX (Average Directional Index)."""
    # 1. TR
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # 2. DM
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)
    
    # 3. Smoothed (Wilder's)
    # alpha = 1/window
    # Pandas ewm(com=window-1)
    tr_smooth = tr.ewm(com=window-1, min_periods=window, adjust=False).mean() # Actually Wilder uses Sum for first then smooth? 
    # Standard ADX usually uses Smoothed Moving Average which is equivalent to EMA(alpha=1/N).
    # We will use EMA directly.
    
    plus_di = 100 * (plus_dm.ewm(com=window-1, min_periods=window, adjust=False).mean() / tr_smooth)
    minus_di = 100 * (minus_dm.ewm(com=window-1, min_periods=window, adjust=False).mean() / tr_smooth)
    
    # 4. DX
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
    
    # 5. ADX (Smoothed DX)
    adx = dx.ewm(com=window-1, min_periods=window, adjust=False).mean()
    
    return adx


def _validate_required_columns(df: pd.DataFrame) -> None:
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"缺少必要欄位: {sorted(missing)}")


__all__ = ["StarFeatureCache", "build_feature_frame"]
