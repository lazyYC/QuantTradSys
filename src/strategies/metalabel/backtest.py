"""
Backtest logic for Meta-Labeling Strategy.
Two-layer execution: primary signals filtered by meta-model probability.
"""
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd

from strategies.metalabel.params import (
    PrimarySignalParams, TripleBarrierParams, MetaModelParams
)
from strategies.metalabel.primary import generate_primary_signals
from strategies.metalabel.features import build_meta_features, get_feature_columns
from strategies.metalabel.model import load_meta_model, predict_meta_probability

LOGGER = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Backtest result container."""
    trades: pd.DataFrame
    metrics: Dict[str, float]
    equity_curve: pd.DataFrame


def simulate_trade(
    df: pd.DataFrame,
    entry_idx: int,
    signal: int,
    barrier_params: TripleBarrierParams,
) -> Dict[str, Any]:
    """Simulate a single trade with triple barrier exit."""
    
    entry_bar = df.iloc[entry_idx]
    entry_price = entry_bar["close"]
    entry_time = entry_bar["timestamp"]
    
    # Calculate barriers
    if signal == 1:  # Long
        take_profit = entry_price * (1 + barrier_params.take_profit_pct)
        stop_loss = entry_price * (1 - barrier_params.stop_loss_pct)
    else:  # Short
        take_profit = entry_price * (1 - barrier_params.take_profit_pct)
        stop_loss = entry_price * (1 + barrier_params.stop_loss_pct)
    
    max_exit_idx = min(entry_idx + barrier_params.max_holding_bars, len(df) - 1)
    
    exit_idx = max_exit_idx
    exit_reason = "timeout"
    exit_price = df.iloc[max_exit_idx]["close"]
    
    for i in range(entry_idx + 1, max_exit_idx + 1):
        bar = df.iloc[i]
        
        if signal == 1:  # Long
            if bar["high"] >= take_profit:
                exit_idx = i
                exit_price = take_profit
                exit_reason = "take_profit"
                break
            if bar["low"] <= stop_loss:
                exit_idx = i
                exit_price = stop_loss
                exit_reason = "stop_loss"
                break
        else:  # Short
            if bar["low"] <= take_profit:
                exit_idx = i
                exit_price = take_profit
                exit_reason = "take_profit"
                break
            if bar["high"] >= stop_loss:
                exit_idx = i
                exit_price = stop_loss
                exit_reason = "stop_loss"
                break
    
    exit_bar = df.iloc[exit_idx]
    exit_time = exit_bar["timestamp"]
    
    # Calculate return
    if signal == 1:
        trade_return = (exit_price - entry_price) / entry_price
    else:
        trade_return = (entry_price - exit_price) / entry_price
    
    holding_mins = (exit_time - entry_time).total_seconds() / 60.0
    
    return {
        "side": "LONG" if signal == 1 else "SHORT",
        "entry_time": entry_time,
        "exit_time": exit_time,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "return": trade_return,
        "holding_mins": holding_mins,
        "exit_reason": exit_reason,
        "entry_zscore": 0.0,
        "exit_zscore": 0.0,
    }


def run_backtest(
    df: pd.DataFrame,
    model_path: Optional[str],
    primary_params: PrimarySignalParams,
    barrier_params: TripleBarrierParams,
    meta_params: MetaModelParams,
    core_start: Optional[pd.Timestamp] = None,
    transaction_cost: float = 0.001,
) -> BacktestResult:
    """
    Run two-layer backtest.
    
    1. Generate primary signals (high recall)
    2. Filter via meta-model probability
    3. Simulate trades with triple barrier
    """
    LOGGER.info("Starting Meta-Labeling backtest...")
    
    # Ensure sorted by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    
    # Generate primary signals
    signal_df = generate_primary_signals(df, primary_params)
    
    # Build meta features
    meta_features = build_meta_features(df, meta_params)
    
    # Load meta-model if available
    if model_path:
        meta_model, feature_cols, _ = load_meta_model(model_path)
        
        # Predict probabilities for all bars
        meta_proba = predict_meta_probability(meta_model, meta_features, feature_cols)
    else:
        # No model: use all signals (for training phase)
        meta_proba = np.ones(len(df))
    
    # Add meta probability to signal df
    signal_df["meta_proba"] = meta_proba
    
    # Filter by core_start
    if core_start is not None:
        eval_mask = signal_df["timestamp"] >= core_start
        signal_df = signal_df[eval_mask].copy()
    
    # Get signal bars that pass meta-threshold
    signal_mask = (signal_df["signal"] != 0) & (signal_df["meta_proba"] >= meta_params.meta_threshold)
    filtered_signals = signal_df[signal_mask]
    
    LOGGER.info(
        f"Primary signals: {(signal_df['signal'] != 0).sum()}, "
        f"Passed meta-filter: {len(filtered_signals)}"
    )
    
    # Simulate trades
    trades = []
    in_trade_until = -1
    
    for idx, row in filtered_signals.iterrows():
        if idx <= in_trade_until:
            continue  # Skip signals during active trade
        
        trade = simulate_trade(df, idx, row["signal"], barrier_params)
        trade["meta_proba"] = row["meta_proba"]
        trade["signal_type"] = row.get("signal_type", "")
        trades.append(trade)
        
        # Block until this trade exits
        exit_idx = df[df["timestamp"] == trade["exit_time"]].index
        if len(exit_idx) > 0:
            in_trade_until = exit_idx[0]
    
    trades_df = pd.DataFrame(trades)
    
    # Calculate metrics
    if trades_df.empty:
        metrics = {
            "trades": 0,
            "win_rate": 0,
            "total_return": 0,
            "sharpe": 0,
            "max_drawdown": 0,
        }
        equity_curve = pd.DataFrame()
    else:
        # Apply transaction costs
        trades_df["return"] -= transaction_cost * 2  # Entry + exit
        
        wins = (trades_df["return"] > 0).sum()
        total = len(trades_df)
        
        returns = trades_df["return"]
        total_return = (1 + returns).prod() - 1
        
        # Sharpe (annualized, assuming 1min bars)
        if returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 24 * 60 / trades_df["holding_mins"].mean())
        else:
            sharpe = 0
        
        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdowns = (cumulative - running_max) / running_max
        max_dd = drawdowns.min()
        
        metrics = {
            "trades": total,
            "win_rate": wins / total if total > 0 else 0,
            "total_return": total_return,
            "sharpe": sharpe,
            "max_drawdown": abs(max_dd),
            "avg_return": returns.mean(),
            "avg_holding_mins": trades_df["holding_mins"].mean(),
        }
        
        # Build equity curve from trades
        equity_curve = build_equity_curve_from_trades(df, trades_df)
    
    LOGGER.info(f"Backtest complete: {metrics}")
    
    return BacktestResult(
        trades=trades_df,
        metrics=metrics,
        equity_curve=equity_curve,
    )


def build_equity_curve_from_trades(
    df: pd.DataFrame, 
    trades: pd.DataFrame,
    initial_capital: float = 10000.0
) -> pd.DataFrame:
    """Build per-bar equity curve from trades."""
    
    equity = np.full(len(df), initial_capital)
    capital = initial_capital
    
    for _, trade in trades.iterrows():
        entry_time = trade["entry_time"]
        exit_time = trade["exit_time"]
        trade_return = trade["return"]
        
        # Find exit index
        exit_mask = df["timestamp"] == exit_time
        if exit_mask.any():
            exit_idx = df[exit_mask].index[0]
            capital *= (1 + trade_return)
            equity[exit_idx:] = capital
    
    return pd.DataFrame({
        "timestamp": df["timestamp"],
        "equity": equity,
    })
