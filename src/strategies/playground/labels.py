"""star_xgb 策略的標籤建構 (Mean Reversion Dynamic Targets)。"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .params import StarIndicatorParams

# 3 Classes: -1 (Short), 0 (Neutral), 1 (Long)
RETURN_CLASSES = (-1, 0, 1)


def build_label_frame(
    features: pd.DataFrame,
    params: StarIndicatorParams,
    *,
    thresholds: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    根據 Mean Reversion 邏輯建立標籤。
    Target: Trend MA
    Logic: Dynamic Triple Barrier
    """
    required = {"timestamp", "close", "high", "low", "trend_ma"}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"標籤建構缺少欄位: {sorted(missing)}")

    close = features["close"].astype(float)
    high = features["high"].astype(float)
    low = features["low"].astype(float)
    trend_ma = features["trend_ma"].astype(float)

    future_window = max(int(params.future_window), 1)
    
    # 預設參數 (若 params 中無設定則使用預設值)
    # stop_loss_pct 應由 params 提供
    stop_loss_pct = getattr(params, "stop_loss_pct", 0.005)
    # min_profit_threshold 使用 future_return_threshold
    min_profit_pct = params.future_return_threshold
    profit_ratio = 0.6  # 回歸目標的 60% (Relaxed for vector-2.7)

    # Forward Window Calculation
    # 使用 FixedForwardWindowIndexer 來正確計算未來窗口的極值
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=future_window)
    # shift(-1) 因為我們是在當前 bar 結束時做決策，看向未來 N 根 (不含當前)
    # 但 FixedForwardWindow 包含起點。
    # 嚴謹來說：Entry Price = Close.
    # High/Low Check 應從 Next Bar 開始。
    # 所以先 shift(-1) 再 rolling 是正確的。
    future_max_high = high.shift(-1).rolling(window=indexer).max()
    future_min_low = low.shift(-1).rolling(window=indexer).min()

    # Calculate Distance and Targets
    dist = close - trend_ma
    dist_pct = (dist.abs() / close).replace([np.inf, -np.inf], 0.0)

    # 只有當乖離夠大才有交易價值
    valid_mask = dist_pct >= min_profit_pct

    labels = np.zeros(len(close), dtype=int)

    # --- Long Logic (Price < MA) ---
    # 1. 位置判斷: 收盤價在均線下，且乖離夠大
    long_mask = (dist < 0) & valid_mask
    
    # 2. 目標價格
    long_tp = close + dist.abs() * profit_ratio
    long_sl = close * (1.0 - stop_loss_pct)

    # 3. Triple Barrier Check
    # 條件: 未來 N 根內，最高價觸及止盈
    # 且: 最低價未觸及止損 (這裡用保守估計：整個窗口的最低價都必須高於止損)
    # 這比 "先碰止盈還是先碰止損" 更嚴格，但在 Vectorized 下是安全的假設。
    long_success = (future_max_high >= long_tp) & (future_min_low > long_sl)
    
    labels[long_mask & long_success] = 1

    # --- Short Logic (Price > MA) ---
    # 1. 位置判斷: 收盤價在均線上，且乖離夠大
    short_mask = (dist > 0) & valid_mask
    
    # 2. 目標價格
    short_tp = close - dist.abs() * profit_ratio
    short_sl = close * (1.0 + stop_loss_pct)

    # 3. Triple Barrier Check
    # 條件: 未來 N 根內，最低價觸及止盈
    # 且: 最高價未觸及止損
    short_success = (future_min_low <= short_tp) & (future_max_high < short_sl)

    labels[short_mask & short_success] = -1

    # --- Auxiliary Columns ---
    # 為了保持 dataset 結構完整性，保留 future_close_return 計算
    future_close = close.shift(-future_window)
    future_long_return = (future_close - close) / close.replace(0.0, pd.NA)
    future_short_return = -future_long_return
    
    # 這裡的欄位名稱必須與 dataset.py 或 backtest.py 預期的一致
    label_frame = features[["timestamp"]].copy()
    label_frame["future_long_return"] = future_long_return
    label_frame["future_short_return"] = future_short_return
    label_frame["return_class"] = labels # Explicit name matching TARGET_COLUMN usually
    
    # 補上舊欄位以免 break code (雖然 dataset.py 裡 list_feature_columns 排除它們，但 merge 時可能用到)
    label_frame["future_min_return"] = 0.0
    label_frame["future_best_short_return"] = 0.0
    label_frame["future_best_long_return"] = 0.0
    label_frame["future_close_return"] = future_long_return

    return label_frame, {}

__all__ = ["build_label_frame", "RETURN_CLASSES"]
