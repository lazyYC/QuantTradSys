"""
Triple Barrier Method labeling for Meta-Labeling Strategy.
Assigns meta-labels to primary signals based on trade outcomes.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

from strategies.metalabel.params import TripleBarrierParams

LOGGER = logging.getLogger(__name__)


def apply_triple_barrier(
    df: pd.DataFrame,
    signal_indices: pd.Index,
    params: TripleBarrierParams,
) -> pd.DataFrame:
    """
    Apply Triple Barrier Method to each signal.
    
    For each signal bar, determine if the trade would be:
    - Profitable (hit take profit first) -> meta_label = 1
    - Unprofitable (hit stop loss or timeout) -> meta_label = 0
    
    Args:
        df: Full OHLCV DataFrame with 'close', 'high', 'low' columns
        signal_indices: Index of bars where primary signals were generated
        params: Triple barrier parameters
        
    Returns:
        DataFrame with signal_idx, meta_label, exit_bar, exit_reason columns
    """
    results = []
    
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    
    # Calculate ATR if needed
    if params.use_atr:
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - df["close"].shift(1)).abs()
        tr3 = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=params.atr_window).mean().fillna(0).values
    else:
        atr = None
    
    df_index = df.index.tolist()
    
    for sig_idx in signal_indices:
        # Get position in array
        try:
            pos = df_index.index(sig_idx)
        except ValueError:
            continue
        
        entry_price = close[pos]
        signal_value = df.loc[sig_idx, "signal"]
        
        # Calculate barrier prices
        # Calculate barrier prices
        if params.use_atr and atr is not None and atr[pos] > 0:
            volatility = atr[pos]
            tp_dist = volatility * params.tp_atr_mult
            sl_dist = volatility * params.sl_atr_mult
        else:
            tp_dist = entry_price * params.take_profit_pct
            sl_dist = entry_price * params.stop_loss_pct

        # Enforce minimum profit distance (cover transaction costs)
        # If ATR is too low, we must extend TP to at least min_profit_pct
        min_tp_dist = entry_price * params.min_profit_pct
        if tp_dist < min_tp_dist:
            tp_dist = min_tp_dist

        if signal_value == 1:  # Long
            take_profit_price = entry_price + tp_dist
            stop_loss_price = entry_price - sl_dist
        else:  # Short
            take_profit_price = entry_price - tp_dist
            stop_loss_price = entry_price + sl_dist
        
        # Scan forward to find exit
        meta_label = 0  # Default: unprofitable
        exit_bar = None
        exit_reason = "timeout"
        
        max_bar = min(pos + params.max_holding_bars, len(df) - 1)
        
        for bar_idx in range(pos + 1, max_bar + 1):
            if signal_value == 1:  # Long position
                # Check if high hits take profit
                if high[bar_idx] >= take_profit_price:
                    meta_label = 1
                    exit_bar = bar_idx
                    exit_reason = "take_profit"
                    break
                # Check if low hits stop loss
                if low[bar_idx] <= stop_loss_price:
                    meta_label = 0
                    exit_bar = bar_idx
                    exit_reason = "stop_loss"
                    break
            else:  # Short position
                # Check if low hits take profit
                if low[bar_idx] <= take_profit_price:
                    meta_label = 1
                    exit_bar = bar_idx
                    exit_reason = "take_profit"
                    break
                # Check if high hits stop loss
                if high[bar_idx] >= stop_loss_price:
                    meta_label = 0
                    exit_bar = bar_idx
                    exit_reason = "stop_loss"
                    break
        
        if exit_bar is None:
            exit_bar = max_bar
        
        results.append({
            "signal_idx": sig_idx,
            "signal": signal_value,
            "entry_price": entry_price,
            "take_profit_price": take_profit_price,
            "stop_loss_price": stop_loss_price,
            "meta_label": meta_label,
            "exit_bar_pos": exit_bar,
            "exit_reason": exit_reason,
        })
    
    result_df = pd.DataFrame(results)
    
    if not result_df.empty:
        # Log distribution
        label_counts = result_df["meta_label"].value_counts()
        reason_counts = result_df["exit_reason"].value_counts()
        LOGGER.info(
            f"Meta-labels generated: Profitable={label_counts.get(1, 0)}, "
            f"Unprofitable={label_counts.get(0, 0)} | "
            f"Exits: TP={reason_counts.get('take_profit', 0)}, "
            f"SL={reason_counts.get('stop_loss', 0)}, "
            f"Timeout={reason_counts.get('timeout', 0)}"
        )
    
    return result_df


def build_labeled_dataset(
    df: pd.DataFrame,
    signal_df: pd.DataFrame,
    barrier_params: TripleBarrierParams,
) -> pd.DataFrame:
    """
    Build a labeled dataset for meta-model training.
    
    Args:
        df: Full OHLCV data with primary signals
        signal_df: DataFrame with signal bars only
        barrier_params: Triple barrier parameters
        
    Returns:
        DataFrame with signal bars and their meta-labels
    """
    signal_indices = signal_df.index
    
    # Apply triple barrier to get labels
    label_df = apply_triple_barrier(df, signal_indices, barrier_params)
    
    if label_df.empty:
        return pd.DataFrame()
    
    # Merge back with signal data
    label_df = label_df.set_index("signal_idx")
    
    # Join with original signal bars
    result = signal_df.join(label_df[["meta_label", "exit_reason"]], how="inner")
    
    return result
