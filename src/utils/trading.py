
import pandas as pd

def build_equity_from_trades(trades: pd.DataFrame, candles: pd.DataFrame = None) -> pd.DataFrame:
    """
    Build a Mark-to-Market equity curve by simulating position exposure over time.
    Calculates 100% unleveraged equity curve assuming:
    1. Returns are compounded.
    2. Position = sum of active trades (direction * quantity). 
       Assuming quantity=1 per trade (stack) implies leverage if count > 1.
    """
    if candles is None or candles.empty:
        # Fallback to simple closed-trade equity if no candles provided
        return _build_simple_equity(trades)

    if trades.empty:
         return pd.DataFrame({"timestamp": candles["timestamp"], "equity": 1.0})

    # 1. Align times
    # Ensure candles are sorted and unique
    df = candles[["timestamp", "close"]].copy().sort_values("timestamp").drop_duplicates("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    # df["close"] = pd.to_numeric(df["close"], errors="coerce").ffill() 
    # Close is usually float, assuming upstream handled it, but ffill for safety around gaps if any
    df["close"] = df["close"].ffill()

    longs = trades[trades["side"] == "LONG"]
    shorts = trades[trades["side"] == "SHORT"]

    def _agg_counts(frame, sign, col):
        if frame.empty: return None
        times = pd.to_datetime(frame[col], utc=True)
        counts = times.value_counts()
        return counts * sign

    l_in = _agg_counts(longs, 1, "entry_time")
    l_out = _agg_counts(longs, -1, "exit_time")
    s_in = _agg_counts(shorts, -1, "entry_time")
    s_out = _agg_counts(shorts, 1, "exit_time")

    all_changes = pd.concat([l_in, l_out, s_in, s_out], axis=0)
    # Handle all None
    if all_changes.empty:
         return pd.DataFrame({"timestamp": candles["timestamp"], "equity": 1.0})
         
    all_changes = all_changes.groupby(level=0).sum()
    
    # Reindex to candle timestamps
    aligned_changes = all_changes.reindex(df.index, fill_value=0)
    
    # Position at End of Bar T = Cumulative Sum of changes up to T
    current_pos = aligned_changes.cumsum()
    
    # Exposure for Bar T (Return T-1 to T) is determined by Position at T-1
    exposure = current_pos.shift(1).fillna(0)
    
    # 3. Calculate Returns
    price_ret = df["close"].pct_change().fillna(0)
    
    # Strategy Return = Exposure * Asset Return
    strat_ret = exposure * price_ret
    
    # 4. Transaction Costs?
    cost_rate = 0.001 
    volume = aligned_changes.abs()
    costs = volume * cost_rate
    
    total_ret = strat_ret - costs
    
    # 5. Equity Curve
    equity = (1 + total_ret).cumprod()
    
    return pd.DataFrame({"timestamp": equity.index, "equity": equity.values})


def _build_simple_equity(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or "return" not in trades.columns:
        return pd.DataFrame(columns=["timestamp", "equity"])
    
    closed = trades.dropna(subset=["exit_time", "return"]).copy()
    if closed.empty:
        return pd.DataFrame(columns=["timestamp", "equity"])
    
    # Validate return column is a proper Series
    return_col = closed["return"]
    if not isinstance(return_col, pd.Series) or len(return_col) == 0:
        return pd.DataFrame(columns=["timestamp", "equity"])
        
    closed = closed.sort_values("exit_time")
    returns = pd.to_numeric(closed["return"], errors="coerce").fillna(0.0)
    equity_values = (1 + returns).cumprod()
    
    return pd.DataFrame({"timestamp": closed["exit_time"], "equity": equity_values})
