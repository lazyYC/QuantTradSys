"""star_xgb 策略即時推論模組。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from utils.formatting import round_numeric_fields

from .features import StarFeatureCache
from .model import CLASS_VALUES
from .params import StarIndicatorParams, StarModelParams


@dataclass
class StarRuntimeState:
    last_signal_timestamp: Optional[pd.Timestamp] = None
    position_side: Optional[str] = None
    entry_price: Optional[float] = None
    entry_timestamp: Optional[pd.Timestamp] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "last_signal_timestamp": self.last_signal_timestamp.isoformat()
            if self.last_signal_timestamp
            else None,
            "position_side": self.position_side,
            "entry_price": self.entry_price,
            "entry_timestamp": self.entry_timestamp.isoformat()
            if self.entry_timestamp
            else None,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, object]]) -> "StarRuntimeState":
        if not data:
            return cls()
        last_ts = data.get("last_signal_timestamp")
        entry_ts_raw = data.get("entry_timestamp")
        timestamp = (
            pd.to_datetime(last_ts, utc=True, errors="coerce") if last_ts else None
        )
        entry_timestamp = (
            pd.to_datetime(entry_ts_raw, utc=True, errors="coerce")
            if entry_ts_raw
            else None
        )
        entry_price = data.get("entry_price")
        if isinstance(entry_price, str):
            try:
                entry_price = float(entry_price)
            except ValueError:
                entry_price = None
        return cls(
            last_signal_timestamp=timestamp,
            position_side=data.get("position_side"),
            entry_price=entry_price,
            entry_timestamp=entry_timestamp,
        )


def generate_realtime_signal(
    df: pd.DataFrame,
    indicator_params: StarIndicatorParams,
    model_params: StarModelParams,
    model: lgb.Booster,
    feature_columns: Sequence[str],
    class_means: Sequence[float],
    cache: Optional[StarFeatureCache] = None,
    state: Optional[StarRuntimeState] = None,
) -> Tuple[str, Dict[str, object], StarRuntimeState]:
    """根據最新 K 線資料產出 star_xgb 策略的即時訊號。"""
    if df.empty:
        return "HOLD", {"reason": "no_data"}, state or StarRuntimeState()

    runtime_state = state or StarRuntimeState()
    if not class_means:
        return "HOLD", {"reason": "missing_class_means"}, runtime_state

    cache = cache or _build_cache_for_runtime(df, indicator_params)
    features = cache.build_features(indicator_params, df)
    if features.empty:
        return "HOLD", {"reason": "insufficient_features"}, runtime_state

    latest = features.iloc[-1]
    feature_vector = latest.reindex(feature_columns)
    feature_vector = pd.to_numeric(feature_vector, errors="coerce").fillna(0.0)
    probs = model.predict(feature_vector.to_numpy(dtype=float).reshape(1, -1))

    class_means_arr = np.asarray(class_means, dtype=float)
    if class_means_arr.size != len(CLASS_VALUES):
        padded = [
            class_means[i] if i < len(class_means) else 0.0
            for i in range(len(CLASS_VALUES))
        ]
        class_means_arr = np.asarray(padded, dtype=float)

    expected_return = _expected_return(probs, class_means_arr)
    pred_class = CLASS_VALUES[int(np.argmax(probs))]

    timestamp = pd.to_datetime(latest["timestamp"], utc=True, errors="coerce")
    price = float(latest.get("close", np.nan))
    threshold = float(model_params.decision_threshold)

    context: Dict[str, object] = {
        "timestamp": str(latest["timestamp"]),
        "price": price,
        "expected_return": expected_return,
        "predicted_class": int(pred_class),
    }

    if runtime_state.position_side:
        exit_signal = False
        if runtime_state.position_side == "LONG":
            exit_signal = pred_class in {0, -1, -2}
        elif runtime_state.position_side == "SHORT":
            exit_signal = pred_class in {0, 1, 2}

        if exit_signal and timestamp is not None and not np.isnan(price):
            if runtime_state.entry_price:
                if runtime_state.position_side == "LONG":
                    trade_return = (
                        price - runtime_state.entry_price
                    ) / runtime_state.entry_price
                else:
                    trade_return = (
                        runtime_state.entry_price - price
                    ) / runtime_state.entry_price
            else:
                trade_return = None
            context.update(
                {
                    "action": f"exit_{runtime_state.position_side.lower()}",
                    "threshold": threshold,
                    "return": trade_return,
                }
            )
            new_state = StarRuntimeState(last_signal_timestamp=timestamp)
            context = round_numeric_fields(context, decimals_map={"expected_return": 4})
            return f"EXIT_{runtime_state.position_side}", context, new_state

    if (
        runtime_state.position_side is None
        and expected_return >= threshold
        and not np.isnan(price)
    ):
        if pred_class == 2:
            context.update({"action": "enter_long", "threshold": threshold})
            new_state = StarRuntimeState(
                last_signal_timestamp=timestamp,
                position_side="LONG",
                entry_price=price,
                entry_timestamp=timestamp,
            )
            context = round_numeric_fields(context, decimals_map={"expected_return": 4})
            return "ENTER_LONG", context, new_state
        if pred_class == -2:
            context.update({"action": "enter_short", "threshold": threshold})
            new_state = StarRuntimeState(
                last_signal_timestamp=timestamp,
                position_side="SHORT",
                entry_price=price,
                entry_timestamp=timestamp,
            )
            context = round_numeric_fields(context, decimals_map={"expected_return": 4})
            return "ENTER_SHORT", context, new_state

    context.update({"reason": "no_action", "threshold": threshold})
    context = round_numeric_fields(context, decimals_map={"expected_return": 4})
    return "HOLD", context, runtime_state


def load_star_model(model_path: Path | str) -> lgb.Booster:
    """載入 LightGBM Booster 供即時推論使用。"""
    return lgb.Booster(model_file=str(model_path))


def _build_cache_for_runtime(
    df: pd.DataFrame, params: StarIndicatorParams
) -> StarFeatureCache:
    return StarFeatureCache(
        df,
        trend_windows=[params.trend_window],
        atr_windows=[params.atr_window],
        volatility_windows=[params.volatility_window],
        volume_windows=[params.volume_window],
        pattern_windows=[params.pattern_lookback],
    )


def _expected_return(probs: np.ndarray, class_means: np.ndarray) -> float:
    if probs.ndim == 1:
        return float(probs @ class_means)
    return float(probs.reshape(-1, len(class_means)) @ class_means)


__all__ = [
    "StarRuntimeState",
    "generate_realtime_signal",
    "load_star_model",
]
