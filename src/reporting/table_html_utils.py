
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from reporting.plotting import build_scatter_figure

def generate_metrics_html(metrics_df: pd.DataFrame) -> str:
    """Generate HTML table for metrics."""
    if metrics_df.empty:
        return ""
        
    # Transpose for better view if it's a single row
    m_df = metrics_df.copy()
    if len(m_df) == 1:
        m_df = m_df.T.reset_index()
        m_df.columns = ["Metric", "Value"]
    
    return m_df.to_html(classes="data-table", index=False, border=0, float_format=lambda x: f"{x:.4f}")

def generate_distribution_html(trades_df: pd.DataFrame) -> str:
    """Generate HTML table for trade distribution."""
    if trades_df.empty:
        return ""

    t_df = trades_df.copy()
    
    dist_data = {
        "Total Trades": len(t_df),
        "Longs": len(t_df[t_df["side"] == "LONG"]) if "side" in t_df.columns else 0,
        "Shorts": len(t_df[t_df["side"] == "SHORT"]) if "side" in t_df.columns else 0,
        "Win Rate": f"{(len(t_df[t_df['return'] > 0]) / len(t_df) * 100):.1f}%" if len(t_df) > 0 and "return" in t_df.columns else "0%",
        "Avg Return": f"{t_df['return'].mean():.4f}" if "return" in t_df.columns and len(t_df) > 0 else "0.0000",
    }
    
    if "holding_mins" in t_df.columns and len(t_df) > 0:
        try:
            # Ensure it's a Series before calling pd.to_numeric
            hold_col = t_df["holding_mins"]
            if isinstance(hold_col, pd.Series) and len(hold_col) > 0:
                hold_mins = pd.to_numeric(hold_col, errors='coerce').fillna(0)
                dist_data["Holding Avg (m)"] = f"{hold_mins.mean():.1f}"
                dist_data["Holding Median (m)"] = f"{hold_mins.median():.1f}"
                dist_data["Holding Max (m)"] = f"{hold_mins.max():.1f}"
        except Exception:
            pass  # Silently skip if holding_mins processing fails
    
    dist_df = pd.DataFrame([dist_data])
    dist_df = dist_df.T.reset_index()
    dist_df.columns = ["Stat", "Value"]
    
    return dist_df.to_html(classes="data-table", index=False, border=0)

def generate_analysis_html(trades_df: pd.DataFrame) -> str:
    """Generate HTML for correlation analysis and scatter plot."""
    if trades_df.empty:
        return ""

    t_df = trades_df.copy()
    analysis_html = ""
    
    if "holding_mins" in t_df.columns and len(t_df) > 0:
        try:
            # Ensure columns are Series before calling pd.to_numeric
            return_col = t_df["return"] if "return" in t_df.columns else pd.Series([0.0] * len(t_df))
            hold_col = t_df["holding_mins"]
            
            if not isinstance(return_col, pd.Series) or not isinstance(hold_col, pd.Series):
                return ""
            
            if len(hold_col) == 0:
                return ""
                
            t_df["return_numeric"] = pd.to_numeric(return_col, errors='coerce').fillna(0)
            t_df["hold_numeric"] = pd.to_numeric(hold_col, errors='coerce').fillna(0)
            
            # Safe correlation (requires > 1 point)
            corr = t_df["hold_numeric"].corr(t_df["return_numeric"]) if len(t_df) > 1 else 0.0
            
            long_mask = t_df["side"] == "LONG" if "side" in t_df.columns else pd.Series([False] * len(t_df))
            short_mask = t_df["side"] == "SHORT" if "side" in t_df.columns else pd.Series([False] * len(t_df))
            
            l_df = t_df.loc[long_mask]
            s_df = t_df.loc[short_mask]
            
            l_corr = l_df["hold_numeric"].corr(l_df["return_numeric"]) if len(l_df) > 1 else 0.0
            s_corr = s_df["hold_numeric"].corr(s_df["return_numeric"]) if len(s_df) > 1 else 0.0
            
            analysis_data = {
                "Corr (Time vs Return) All": f"{corr:.4f}" if not pd.isna(corr) else "0.0000",
                "Corr (Time vs Return) Long": f"{l_corr:.4f}" if not pd.isna(l_corr) else "0.0000",
                "Corr (Time vs Return) Short": f"{s_corr:.4f}" if not pd.isna(s_corr) else "0.0000",
            }
            analysis_df = pd.DataFrame([analysis_data]).T.reset_index()
            analysis_df.columns = ["Metric", "Value"]
            analysis_html = analysis_df.to_html(classes="data-table", index=False, border=0)
            
            # Add Scatter Plot
            scatter_fig = build_scatter_figure(t_df, title="Holding Time vs Return")
            if scatter_fig:
                scatter_html = scatter_fig.to_html(full_html=False, include_plotlyjs='cdn')
                analysis_html += f"<div style='margin-top: 20px;'>{scatter_html}</div>"
        except Exception:
            pass  # Silently skip if analysis processing fails
            
    return analysis_html

def generate_trades_html(trades_df: pd.DataFrame) -> str:
    """Generate HTML table for full trade list with navigation links."""
    if trades_df.empty:
        return ""
        
    t_df = trades_df.copy()
    
    # Sort by time
    t_df = t_df.sort_values("entry_time", ascending=False)
    
    # Create Navigation Links for Times
    # Entry Time Link
    t_df["ts_entry"] = pd.to_datetime(t_df["entry_time"], utc=True).astype('int64') // 10**9
    t_df["entry_time_display"] = pd.to_datetime(t_df["entry_time"], utc=True).dt.strftime("%Y-%m-%d %H:%M")
    t_df["entry_time"] = t_df.apply(
        lambda row: f"<span class='time-link' onclick='window.scrollToTimestamp({row['ts_entry']})'>{row['entry_time_display']}</span>", 
        axis=1
    )
    
    # Exit Time Link
    if "exit_time" in t_df.columns:
        t_df["ts_exit"] = pd.to_datetime(t_df["exit_time"], utc=True).astype('int64') // 10**9
        t_df["exit_time_display"] = pd.to_datetime(t_df["exit_time"], utc=True).dt.strftime("%Y-%m-%d %H:%M")
        t_df["exit_time"] = t_df.apply(
            lambda row: f"<span class='time-link' onclick='window.scrollToTimestamp({row['ts_exit']})'>{row['exit_time_display']}</span>", 
            axis=1
        )

    # Format columns
    view_cols = ["entry_time", "exit_time", "side", "entry_price", "exit_price", "return", "holding_mins", "exit_reason"]
    # Ensure columns exist
    view_cols = [c for c in view_cols if c in t_df.columns]
    
    view_df = t_df[view_cols].copy()
    # Note: Dates are already formatted as HTML strings above
    
    if "return" in view_df.columns:
        view_df["return"] = view_df["return"].map(lambda x: f"{float(x):.4f}" if pd.notna(x) else "-")
    if "holding_mins" in view_df.columns:
        view_df["holding_mins"] = view_df["holding_mins"].map(lambda x: f"{float(x):.1f}" if pd.notna(x) else "-")
    if "entry_price" in view_df.columns:
        view_df["entry_price"] = view_df["entry_price"].map(lambda x: f"{float(x):.2f}" if pd.notna(x) else "-")
    if "exit_price" in view_df.columns:
        view_df["exit_price"] = view_df["exit_price"].map(lambda x: f"{float(x):.2f}" if pd.notna(x) else "-")

    return view_df.to_html(classes="data-table", index=False, border=0, escape=False)
