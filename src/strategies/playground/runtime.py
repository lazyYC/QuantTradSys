"""star_xgb 策略即時推論模組。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from utils.formatting import round_numeric_fields

from .dataset import apply_trade_amount_scaling
from .features import StarFeatureCache
from .model import CLASS_VALUES
from .params import StarIndicatorParams, StarModelParams


@dataclass
class StarRuntimeState:
    last_signal_timestamp: Optional[pd.Timestamp] = None
    position_side: Optional[str] = None
    entry_price: Optional[float] = None
    entry_timestamp: Optional[pd.Timestamp] = None
    min_exit_timestamp: Optional[pd.Timestamp] = None
    # New: Track active stacks for pyramiding
    stacks: Optional[list[dict]] = None

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
            "min_exit_timestamp": self.min_exit_timestamp.isoformat()
            if self.min_exit_timestamp
            else None,
            "stacks": self.stacks,
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
        min_exit_raw = data.get("min_exit_timestamp")
        min_exit_timestamp = (
            pd.to_datetime(min_exit_raw, utc=True, errors="coerce")
            if min_exit_raw
            else None
        )
        return cls(
            last_signal_timestamp=timestamp,
            position_side=data.get("position_side"),
            entry_price=entry_price,
            entry_timestamp=entry_timestamp,
            min_exit_timestamp=min_exit_timestamp,
            stacks=data.get("stacks"),
        )


def generate_realtime_signal(
    df: pd.DataFrame,
    indicator_params: StarIndicatorParams,
    model_params: StarModelParams,
    model: lgb.Booster,
    feature_columns: Sequence[str],
    class_means: Sequence[float],
    feature_stats: Optional[Dict[str, Dict[str, float]]] = None,
    cache: Optional[StarFeatureCache] = None,
    state: Optional[StarRuntimeState] = None,
    min_hold_duration: Optional[pd.Timedelta] = None,
    stop_loss_pct: Optional[float] = None,
) -> Tuple[str, Dict[str, object], StarRuntimeState]:
    """根據最新 K 線資料產出 star_xgb 策略的即時訊號 (支援 Pyramiding)。"""
    if df.empty:
        return "HOLD", {"reason": "no_data"}, state or StarRuntimeState()

    runtime_state = state or StarRuntimeState()
    if not class_means:
        return "HOLD", {"reason": "missing_class_means"}, runtime_state

    # Initialize stacks if None
    if runtime_state.stacks is None:
        runtime_state.stacks = []
        # Migration: if legacy state exists, convert to 1 stack
        if runtime_state.position_side and runtime_state.entry_timestamp:
             runtime_state.stacks.append({
                 "entry_timestamp": runtime_state.entry_timestamp.isoformat(),
                 "entry_price": runtime_state.entry_price,
                 "min_exit_timestamp": runtime_state.min_exit_timestamp.isoformat() if runtime_state.min_exit_timestamp else None,
             })

    cache = cache or _build_cache_for_runtime(df, indicator_params)
    features = cache.build_features(indicator_params, df)
    trade_stats = feature_stats.get("trade_amount") if feature_stats else None
    features = apply_trade_amount_scaling(features, trade_stats)
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

    target_price = None
    if not np.isnan(price) and np.isfinite(price):
        if pred_class == 2:
            target_price = price * (1.0 + expected_return)
        elif pred_class == -2:
            target_price = price * (1.0 - expected_return)

    context: Dict[str, object] = {
        "time_utc": str(latest["timestamp"]),
        "signal_price": price,
        "expected_return": expected_return,
        "predicted_class": int(pred_class),
        "target_price": target_price,
        "active_stacks": len(runtime_state.stacks),
    }

    # 1. Check Exits for existing stacks
    stacks_to_remove = []
    longs_to_exit = 0
    shorts_to_exit = 0
    stop_triggered_stacks = 0
    
    for i, stack in enumerate(runtime_state.stacks):
        entry_ts = pd.to_datetime(stack["entry_timestamp"], utc=True)
        entry_price = stack["entry_price"]
        stack_side = stack.get("side", runtime_state.position_side) # Fallback for legacy
        min_exit_ts = pd.to_datetime(stack["min_exit_timestamp"], utc=True) if stack.get("min_exit_timestamp") else None
        
        # Check holding time
        hold_elapsed = False
        if min_exit_ts and timestamp >= min_exit_ts:
            hold_elapsed = True
            
        # Check Stop Loss
        stop_triggered = False
        if entry_price and not np.isnan(price):
            if stack_side == "LONG":
                trade_return = (price - entry_price) / entry_price
                if stop_loss_pct and trade_return <= -abs(stop_loss_pct):
                    stop_triggered = True
            elif stack_side == "SHORT":
                trade_return = (entry_price - price) / entry_price
                if stop_loss_pct and trade_return <= -abs(stop_loss_pct):
                    stop_triggered = True
        
        should_exit = False
        if stop_triggered:
            should_exit = True
            stop_triggered_stacks += 1
        elif hold_elapsed:
            should_exit = True
            
        if should_exit:
            stacks_to_remove.append(i)
            if stack_side == "LONG":
                longs_to_exit += 1
            elif stack_side == "SHORT":
                shorts_to_exit += 1

    # Remove exited stacks
    if stacks_to_remove:
        for i in sorted(stacks_to_remove, reverse=True):
            runtime_state.stacks.pop(i)

    # 2. Check Entries (if not max stacks)
    MAX_STACKS = 5
    entry_action = None
    entry_side = None
    
    if len(runtime_state.stacks) < MAX_STACKS:
        # Check Long Entry
        if pred_class == 2 and expected_return >= threshold:
             entry_action = "ENTER_LONG"
             entry_side = "LONG"
        # Check Short Entry
        elif pred_class == -2 and expected_return >= threshold:
             entry_action = "ENTER_SHORT"
             entry_side = "SHORT"
             
        if entry_action:
             min_exit_timestamp = None
             if min_hold_duration and timestamp is not None:
                min_exit_timestamp = timestamp + min_hold_duration
                
             new_stack = {
                 "entry_timestamp": timestamp.isoformat(),
                 "entry_price": price,
                 "min_exit_timestamp": min_exit_timestamp.isoformat() if min_exit_timestamp else None,
                 "side": entry_side
             }
             runtime_state.stacks.append(new_stack)

    # 3. Calculate Net Action
    # Net Change = (New Longs - Exiting Longs) - (New Shorts - Exiting Shorts)
    # Actually, easier:
    # Action needed:
    # Sell = Exiting Longs + Entering Shorts
    # Buy = Exiting Shorts + Entering Longs
    # Net = Buy - Sell
    
    sells_needed = longs_to_exit + (1 if entry_action == "ENTER_SHORT" else 0)
    buys_needed = shorts_to_exit + (1 if entry_action == "ENTER_LONG" else 0)
    
    net_qty = buys_needed - sells_needed
    
    # Update legacy fields (approximate)
    if not runtime_state.stacks:
        runtime_state.position_side = None
    else:
        # Set to side of latest stack or dominant?
        # Just use latest for simplicity
        runtime_state.position_side = runtime_state.stacks[-1].get("side")
        runtime_state.entry_price = runtime_state.stacks[-1].get("entry_price")
        runtime_state.entry_timestamp = pd.Timestamp(runtime_state.stacks[-1].get("entry_timestamp"), tz="UTC")
        min_exit_str = runtime_state.stacks[-1].get("min_exit_timestamp")
        runtime_state.min_exit_timestamp = pd.Timestamp(min_exit_str, tz="UTC") if min_exit_str else None

    if net_qty == 0:
        context.update({"reason": "no_action_or_net_zero", "threshold": threshold})
        return "HOLD", context, runtime_state
        
    final_action = "ENTER_LONG" if net_qty > 0 else "ENTER_SHORT"
    scale = abs(net_qty)
    
    context.update({
        "action": final_action.lower(),
        "scale": float(scale),
        "reason": "netting_logic",
        "longs_exited": longs_to_exit,
        "shorts_exited": shorts_to_exit,
        "entry_action": entry_action,
        "active_stacks": len(runtime_state.stacks)
    })
    
    new_state = StarRuntimeState(
        last_signal_timestamp=timestamp,
        position_side=runtime_state.position_side,
        entry_price=runtime_state.entry_price,
        entry_timestamp=runtime_state.entry_timestamp,
        min_exit_timestamp=runtime_state.min_exit_timestamp,
        stacks=runtime_state.stacks
    )
    return final_action, context, new_state


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
