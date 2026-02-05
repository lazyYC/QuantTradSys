"""star_xgb 策略的標籤建構 (Binary Regime Classification: Safe vs Unsafe)."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .params import StarIndicatorParams

# 3 Classes: 0 (Noise/Range), 1 (Bull Trend), 2 (Bear Trend)
RETURN_CLASSES = (0, 1, 2)


def build_label_frame(
    features: pd.DataFrame,
    params: StarIndicatorParams,
    *,
    thresholds: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    根據 Directional & Clean Move 邏輯建立標籤。
    Target: 3-Class Directional Trend
    Logic:
        - Class 0 (Noise): 震盪、盤整或擴張形雙巴盤 (Up & Down both high)。
        - Class 1 (Bull): MaxUpside > Threshold AND MaxDownside < StopLoss. (Clean Uptrend)
        - Class 2 (Bear): MaxDownside > Threshold AND MaxUpside < StopLoss. (Clean Downtrend)
        
    Rationale:
        我們只學習「乾淨的趨勢」。如果一個 Window 內同時觸發止盈與止損 (Expanded Volatility)，
        依靠簡單規則(BB)極易被掃出場，因此歸類為 Noise (Class 0)，不予交易。
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

    # Use Stop Loss as the invalidation barrier
    # Default to 0.5% if not specified, or use a reasonable fraction of target if preferred.
    # Here we stick to the params or a default safety net.
    stop_loss = params.stop_loss_pct if params.stop_loss_pct and params.stop_loss_pct > 0 else 0.005

    # Forward Window Calculation
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=future_window)
    
    future_max_high = high.shift(-1).rolling(window=indexer).max()
    future_min_low = low.shift(-1).rolling(window=indexer).min()

    max_upside = (future_max_high - close) / close
    max_downside = (close - future_min_low) / close

    # Initialize all as 0 (Noise)
    labels = np.zeros(len(close), dtype=int)
    
    # Class 1: Bull (Strong Upside, Safe Downside)
    # Note: Using strict inequality < stop_loss to ensure we survive the noise
    is_bull = (max_upside > vol_threshold) & (max_downside < stop_loss)
    
    # Class 2: Bear (Strong Downside, Safe Upside)
    is_bear = (max_downside > vol_threshold) & (max_upside < stop_loss)
    
    # Assign labels
    labels[is_bull] = 1
    labels[is_bear] = 2
    
    # Note: If both conditions met (rare/impossible due to < stop_loss check usually being smaller than threshold),
    # the later assignment wins. But logically:
    # If StopLoss < Threshold, it's impossible to satisfy both Upside > Thresh AND Upside < SL simultaneously.
    # So overlapping is physically impossible if SL < Threshold. 
    # If SL > Threshold, overlap is possible (Big Move Up AND Big Move Down). 
    # In that case, we treat it as Noise (don't overwrite? or overwrite?).
    # My logic above: if both true, Bear wins. But effectively if SL is minimal, neither is true.
    # Let's explicitly handle the "Both High" case as Noise (0).
    
    if stop_loss >= vol_threshold:
        # Conflict possible on expanding chop. Force 0.
        conflict = is_bull & is_bear
        labels[conflict] = 0

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
