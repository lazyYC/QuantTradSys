"""
Meta-Model Features for Meta-Labeling Strategy.
Features designed to be different from primary signals to avoid redundancy.
"""
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from strategies.metalabel.params import MetaModelParams

LOGGER = logging.getLogger(__name__)


@dataclass
class MetaFeatureCache:
    """Cache for pre-computed meta features."""
    features: pd.DataFrame
    
    def __post_init__(self):
        if self.features.empty:
            raise ValueError("MetaFeatureCache cannot be initialized with empty DataFrame")


def compute_volatility_regime(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Compute volatility regime using ATR percentile.
    High value = high volatility regime.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=window, min_periods=1).mean()
    
    # Normalize to percentile over lookback
    atr_pct = atr.rolling(window=window * 5, min_periods=1).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-8) if len(x) > 1 else 0.5,
        raw=False
    )
    return atr_pct.fillna(0.5)


def compute_trend_strength(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Compute ADX for trend strength."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    
    # +DM and -DM
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Smoothed averages
    atr = true_range.rolling(window=window, min_periods=1).mean()
    plus_di = 100 * (plus_dm.rolling(window=window, min_periods=1).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=window, min_periods=1).mean() / atr)
    
    # DX and ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8)
    adx = dx.rolling(window=window, min_periods=1).mean()
    
    return adx.fillna(0)


def compute_volume_anomaly(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Compute volume relative to moving average."""
    volume = df["volume"]
    vol_ma = volume.rolling(window=window, min_periods=1).mean()
    return (volume / vol_ma.replace(0, np.nan)).fillna(1)


def compute_price_autocorrelation(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Compute lag-1 return autocorrelation over rolling window."""
    returns = df["close"].pct_change()
    returns_lag = returns.shift(1)
    
    autocorr = returns.rolling(window=window, min_periods=5).corr(returns_lag)
    return autocorr.fillna(0)


def compute_market_efficiency(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Compute market efficiency ratio.
    High value = trending, Low value = mean-reverting.
    """
    close = df["close"]
    
    # Net change over window
    net_change = (close - close.shift(window)).abs()
    
    # Sum of absolute bar-by-bar changes
    total_change = close.diff().abs().rolling(window=window, min_periods=1).sum()
    
    efficiency = net_change / total_change.replace(0, np.nan)
    return efficiency.fillna(0.5)


def compute_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cyclic time-of-day features."""
    ts = pd.to_datetime(df["timestamp"])
    
    hour = ts.dt.hour + ts.dt.minute / 60.0
    
    # Cyclic encoding
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    
    return pd.DataFrame({
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
    }, index=df.index)


def compute_price_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """Compute various momentum indicators."""
    close = df["close"]
    
    return pd.DataFrame({
        "return_5": close.pct_change(5).fillna(0),
        "return_15": close.pct_change(15).fillna(0),
        "return_60": close.pct_change(60).fillna(0),
    }, index=df.index)


def build_meta_features(
    df: pd.DataFrame, 
    params: MetaModelParams
) -> pd.DataFrame:
    """
    Build all meta-model features.
    
    Args:
        df: OHLCV DataFrame with timestamp column
        params: Meta model parameters
        
    Returns:
        DataFrame with all meta features
    """
    LOGGER.info("Building meta-model features...")
    
    features = pd.DataFrame(index=df.index)
    
    # Volatility regime
    features["volatility_regime"] = compute_volatility_regime(df, params.volatility_window)
    
    # Trend strength (ADX)
    features["trend_strength"] = compute_trend_strength(df)
    
    # Volume anomaly
    features["volume_anomaly"] = compute_volume_anomaly(df, params.volume_window)
    
    # Price autocorrelation
    features["price_autocorr"] = compute_price_autocorrelation(df)
    
    # Market efficiency
    features["market_efficiency"] = compute_market_efficiency(df)
    
    # Time features
    time_feats = compute_time_features(df)
    features = features.join(time_feats)
    
    # Momentum features
    momentum_feats = compute_price_momentum(df)
    features = features.join(momentum_feats)
    
    # Fill any remaining NaNs
    features = features.fillna(0)
    
    LOGGER.info(f"Meta features built: {list(features.columns)}")
    
    return features


def get_feature_columns() -> list:
    """Return list of meta feature column names."""
    return [
        "volatility_regime",
        "trend_strength",
        "volume_anomaly",
        "price_autocorr",
        "market_efficiency",
        "hour_sin",
        "hour_cos",
        "return_5",
        "return_15",
        "return_60",
    ]
