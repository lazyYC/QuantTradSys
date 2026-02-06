"""
Primary Signal Generator (Layer 1) for Meta-Labeling Strategy.
Generates high-recall signals that catch all potential opportunities.
"""
import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

from strategies.metalabel.params import PrimarySignalParams

LOGGER = logging.getLogger(__name__)


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Compute RSI indicator."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_ma_cross(close: pd.Series, fast_window: int, slow_window: int) -> Tuple[pd.Series, pd.Series]:
    """Compute fast and slow moving averages."""
    fast_ma = close.rolling(window=fast_window, min_periods=1).mean()
    slow_ma = close.rolling(window=slow_window, min_periods=1).mean()
    return fast_ma, slow_ma


def compute_bollinger_bands(
    close: pd.Series, window: int = 20, num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Compute Bollinger Bands (middle, upper, lower)."""
    middle = close.rolling(window=window, min_periods=1).mean()
    std = close.rolling(window=window, min_periods=1).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return middle, upper, lower


def generate_primary_signals(
    df: pd.DataFrame, params: PrimarySignalParams
) -> pd.DataFrame:
    """
    Generate primary trading signals with high recall.
    
    Returns DataFrame with columns:
    - signal: 1 (long), -1 (short), 0 (no signal)
    - signal_type: Description of which indicator triggered
    """
    close = df["close"]
    result = df.copy()
    
    # RSI signals
    rsi = compute_rsi(close, params.rsi_window)
    result["rsi"] = rsi
    
    rsi_long = rsi < params.rsi_oversold
    rsi_short = rsi > params.rsi_overbought
    
    # MA Cross signals
    fast_ma, slow_ma = compute_ma_cross(close, params.fast_ma_window, params.slow_ma_window)
    result["fast_ma"] = fast_ma
    result["slow_ma"] = slow_ma
    
    # Detect crossovers
    ma_cross_up = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
    ma_cross_down = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
    
    # Bollinger Band signals
    bb_mid, bb_upper, bb_lower = compute_bollinger_bands(
        close, params.bb_window, params.bb_std
    )
    result["bb_upper"] = bb_upper
    result["bb_lower"] = bb_lower
    result["bb_mid"] = bb_mid
    
    # Touch detection (price was above/below and now touching)
    bb_touch_lower = (close <= bb_lower) & (close.shift(1) > bb_lower.shift(1))
    bb_touch_upper = (close >= bb_upper) & (close.shift(1) < bb_upper.shift(1))
    
    # Combine signals based on mode
    if params.signal_mode == "any":
        # Any signal triggers (high recall)
        long_signal = rsi_long | ma_cross_up | bb_touch_lower
        short_signal = rsi_short | ma_cross_down | bb_touch_upper
    else:
        # All signals must agree (high precision, but we don't want this for Layer 1)
        long_signal = rsi_long & (fast_ma > slow_ma) & (close <= bb_lower)
        short_signal = rsi_short & (fast_ma < slow_ma) & (close >= bb_upper)
    
    # Build signal column
    result["signal"] = 0
    result.loc[long_signal, "signal"] = 1
    result.loc[short_signal, "signal"] = -1
    
    # Signal type for debugging
    result["signal_type"] = ""
    
    # Tag signal sources
    long_mask = result["signal"] == 1
    short_mask = result["signal"] == -1
    
    result.loc[long_mask & rsi_long, "signal_type"] += "rsi_oversold "
    result.loc[long_mask & ma_cross_up, "signal_type"] += "ma_cross_up "
    result.loc[long_mask & bb_touch_lower, "signal_type"] += "bb_lower "
    
    result.loc[short_mask & rsi_short, "signal_type"] += "rsi_overbought "
    result.loc[short_mask & ma_cross_down, "signal_type"] += "ma_cross_down "
    result.loc[short_mask & bb_touch_upper, "signal_type"] += "bb_upper "
    
    result["signal_type"] = result["signal_type"].str.strip()
    
    # Log signal distribution
    signal_counts = result["signal"].value_counts()
    LOGGER.info(f"Primary signals generated: Long={signal_counts.get(1, 0)}, Short={signal_counts.get(-1, 0)}, None={signal_counts.get(0, 0)}")
    
    return result


def get_signal_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Extract only the bars where a signal was generated."""
    return df[df["signal"] != 0].copy()
