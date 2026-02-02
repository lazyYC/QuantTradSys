"""star_xgb 策略的標籤建構 (Binary Regime Classification: Safe vs Unsafe)."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .params import StarIndicatorParams

# 2 Classes: 0 (Safe/Range), 1 (Unsafe/Trend)
RETURN_CLASSES = (0, 1)


def build_label_frame(
    features: pd.DataFrame,
    params: StarIndicatorParams,
    *,
    thresholds: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    根據 Volatility & Breakout 邏輯建立標籤。
    Target: Binary Unsafe Regime
    Logic:
        - Unsafe (1): 如果在未來窗口內，價格波動(最大回撤或最大漲幅)超過 Stop Loss 閾值。
              這代表「即使做網格也會被 Stop Loss 掃出場」的危險區域。
        - Safe (0): 價格保持在區間內，適合網格收租。
    """
    required = {"timestamp", "close", "high", "low"}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"標籤建構缺少欄位: {sorted(missing)}")

    close = features["close"].astype(float)
    high = features["high"].astype(float)
    low = features["low"].astype(float)

    future_window = max(int(params.future_window), 1)
        
    vol_threshold = params.future_return_threshold 
    if vol_threshold <= 0:
        vol_threshold = 0.01 # Fallback 1%

    # Forward Window Calculation
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=future_window)
    
    future_max_high = high.shift(-1).rolling(window=indexer).max()
    future_min_low = low.shift(-1).rolling(window=indexer).min()

    max_upside = (future_max_high - close) / close
    max_downside = (close - future_min_low) / close

    # Label: Unsafe (1) if upside or downside exceeds threshold
    is_unsafe = (max_upside > vol_threshold) | (max_downside > vol_threshold)
    
    labels = is_unsafe.astype(int).values

    future_close = close.shift(-future_window)
    future_return = (future_close - close) / close.replace(0.0, pd.NA)
    
    label_frame = features[["timestamp"]].copy()
    label_frame["future_return"] = future_return
    label_frame["max_upside"] = max_upside
    label_frame["max_downside"] = max_downside
    label_frame["return_class"] = labels 
    
    # Legacy columns filler
    label_frame["future_long_return"] = future_return
    label_frame["future_short_return"] = -future_return

    return label_frame, {}

__all__ = ["build_label_frame", "RETURN_CLASSES"]
