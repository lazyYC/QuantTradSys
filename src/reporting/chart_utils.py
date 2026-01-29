
import pandas as pd
import numpy as np
from typing import List, Dict, Any

def format_candles(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Format OHLCV DataFrame for Lightweight Charts candlestick series."""
    if df.empty:
        return []
    
    # Ensure standard column names if needed, but assuming standard 'open', 'high', 'low', 'close', 'timestamp'
    # Timestamp expected to be datetime (UTC)
    
    data = []
    for _, row in df.iterrows():
        ts = int(row["timestamp"].timestamp())
        data.append({
            "time": ts,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
        })
    return data

def format_volume(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Format volume for Lightweight Charts histogram series."""
    if df.empty:
        return []
        
    data = []
    for _, row in df.iterrows():
        ts = int(row["timestamp"].timestamp())
        color = "#26a69a" if row["close"] >= row["open"] else "#ef5350"
        data.append({
            "time": ts,
            "value": float(row["volume"]),
            "color": color
        })
    return data

def format_equity(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Format equity curve for Lightweight Charts line series."""
    if df.empty:
        return []
        
    data = []
    for _, row in df.iterrows():
        ts = int(row["timestamp"].timestamp())
        data.append({
            "time": ts,
            "value": float(row["equity"])
        })
    return data

def format_signals(df: pd.DataFrame, signal_col: str = "volatility_score") -> List[Dict[str, Any]]:
    """Format signal (volatility score) for Lightweight Charts."""
    if df.empty or signal_col not in df.columns:
        return []
        
    data = []
    for _, row in df.iterrows():
        ts = int(row["timestamp"].timestamp())
        # Assuming timestamp is present. If signal df comes from backtest, it should have it.
        data.append({
            "time": ts,
            "value": float(row[signal_col])
        })
    return data

def format_markers(trades_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Format trade markers for Lightweight Charts."""
    if trades_df.empty:
        return []
        
    raw_markers = []
    for _, row in trades_df.iterrows():
        # Entry
        entry_ts = int(pd.to_datetime(row["entry_time"]).timestamp())
        raw_markers.append({
            "time": entry_ts,
            "position": "belowBar" if row["side"] == "LONG" else "aboveBar",
            "color": "#2196F3" if row["side"] == "LONG" else "#E91E63",
            "shape": "arrowUp" if row["side"] == "LONG" else "arrowDown",
            "text": f"Entry: {float(row['entry_price']):.2f}",
            "id": f"entry_{entry_ts}" 
        })
        
        # Exit
        if pd.notna(row["exit_time"]):
            exit_ts = int(pd.to_datetime(row["exit_time"]).timestamp())
            raw_markers.append({
                "time": exit_ts,
                "position": "aboveBar" if row["side"] == "LONG" else "belowBar", 
                "color": "#FF9800",
                "shape": "circle", 
                "text": f"Exit: {float(row['exit_price']):.2f}",
                "id": f"exit_{exit_ts}" 
            })
    
    # Group markers to prevent overlap (identical logic to engine.py)
    marker_map = {}
    for m in raw_markers:
        key = (m["time"], m["position"], m["color"], m["shape"])
        if key not in marker_map:
            marker_map[key] = []
        marker_map[key].append(m["text"])
    
    final_markers = []
    for (ts, pos, col, shp), texts in marker_map.items():
        combined_text = " | ".join(texts)
        if len(texts) > 2:
            combined_text = f"{len(texts)} trades | " + " | ".join(texts[:2]) + "..."
            
        final_markers.append({
            "time": ts,
            "position": pos,
            "color": col,
            "shape": shp,
            "text": combined_text
        })
    
    final_markers.sort(key=lambda x: x["time"])
    return final_markers
