
"""star_xgb 策略的標籤建構。"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .params import StarIndicatorParams

RETURN_CLASSES = (-2, -1, 0, 1, 2)


def build_label_frame(
    features: pd.DataFrame,
    params: StarIndicatorParams,
    *,
    thresholds: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """根據特徵資料建立多分類標籤與報酬欄位。"""
    required = {"timestamp", "close", "low", "high", "upper_shadow_ratio", "body_ratio", "volume_ratio"}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"標籤建構缺少欄位: {sorted(missing)}")

    future_window = max(int(params.future_window), 1)
    close = features["close"].astype(float)
    low = features["low"].astype(float)
    high = features["high"].astype(float)

    future_close = close.shift(-future_window)
    future_low = low.shift(-1).rolling(window=future_window, min_periods=1).min()
    future_high = high.shift(-1).rolling(window=future_window, min_periods=1).max()

    future_close_return = (future_close - close) / close.replace(0.0, pd.NA)
    future_min_return = (future_low - close) / close.replace(0.0, pd.NA)
    future_short_return = -future_close_return
    future_best_short_return = -future_min_return
    future_long_return = future_close_return
    future_best_long_return = (future_high - close) / close.replace(0.0, pd.NA)

    if thresholds is None:
        valid_returns = future_long_return.dropna().to_numpy()
        if valid_returns.size == 0:
            thresholds = {"q10": 0.0, "q25": 0.0, "q75": 0.0, "q90": 0.0}
        else:
            q10, q25, q75, q90 = np.quantile(valid_returns, [0.1, 0.25, 0.75, 0.9])
            thresholds = {"q10": float(q10), "q25": float(q25), "q75": float(q75), "q90": float(q90)}
    else:
        thresholds = {
            "q10": float(thresholds.get("q10", 0.0)),
            "q25": float(thresholds.get("q25", 0.0)),
            "q75": float(thresholds.get("q75", 0.0)),
            "q90": float(thresholds.get("q90", 0.0)),
        }

    def _assign_class(value: float) -> int:
        if np.isnan(value):
            return 0
        if value >= thresholds["q90"]:
            return 2
        if value >= thresholds["q75"]:
            return 1
        if value <= thresholds["q10"]:
            return -2
        if value <= thresholds["q25"]:
            return -1
        return 0

    return_class = future_long_return.apply(_assign_class).astype(int)

    label_frame = features[["timestamp"]].copy()
    label_frame["candidate"] = (
        (features["upper_shadow_ratio"] >= params.upper_shadow_min)
        & (features["body_ratio"] <= params.body_ratio_max)
        & (features["volume_ratio"] <= params.volume_ratio_max)
    ).astype(int)
    label_frame["future_close_return"] = future_close_return
    label_frame["future_min_return"] = future_min_return
    label_frame["future_short_return"] = future_short_return
    label_frame["future_best_short_return"] = future_best_short_return
    label_frame["future_long_return"] = future_long_return
    label_frame["future_best_long_return"] = future_best_long_return
    label_frame["return_class"] = return_class
    return label_frame, thresholds


__all__ = ["build_label_frame", "RETURN_CLASSES"]
