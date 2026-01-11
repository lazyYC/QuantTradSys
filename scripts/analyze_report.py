
import sys
import os
import json
import re
import pandas as pd
from datetime import datetime

# Regex to find the JSON injection
# html_content.replace("/*INJECT_TRADELIST*/", trades_html) -> This is HTML table
# markers -> "/*INJECT_MARKERS*/", json.dumps(markers)
# But markers don't have full trade info.
# "/*INJECT_METRICS*/" -> HTML
# Wait, the `engine.py` injects `candle_data`, `equity_data`, `markers`.
# It DOES NOT inject raw `trade_list` as JSON, only as HTML table.
# Checking engine.py...
# It creates `trades_html` table.
# But `analysis.py` is hard.
# However, the markers JSON has: {time, position, color, shape, text (price)}.
# And `equity_data` has timestamps.
# The HTML table: `<table class="... dataframe">...</table>`
# Maybe parsing the HTML table is easier? using pd.read_html.

def analyze_report(file_path):
    print(f"Analyzing {file_path}...")
    
    try:
        tables = pd.read_html(file_path)
        print(f"Found {len(tables)} tables.")
        
        # Usually: 
        # 1. Metrics
        # 2. Dist
        # 3. Analysis (Corr)
        # 4. Trade List (The big one)
        
        # Identify Trade List by columns
        trade_df = None
        for df in tables:
            if "entry_time" in df.columns and "exit_reason" in df.columns:
                trade_df = df
                break
        
        if trade_df is None:
            print("Could not find Trade List table.")
            return

        print(f"Trade Data Loaded: {len(trade_df)} trades.")
        
        # Convert columns
        # entry_time is string "YYYY-MM-DD HH:MM"
        trade_df["entry_time"] = pd.to_datetime(trade_df["entry_time"])
        # exit time? Usually not in the default simple view? 
        # engine.py: view_cols = ["entry_time", "side", "entry_price", "exit_price", "return", "holding_mins", "exit_reason"]
        # It does NOT have exit_time explicitly in the table!
        # But we have `holding_mins`.
        # exit_time ~= entry_time + holding_mins
        
        trade_df["holding_mins"] = pd.to_numeric(trade_df["holding_mins"], errors='coerce')
        trade_df["return"] = pd.to_numeric(trade_df["return"], errors='coerce')
        
        # Analysis 1: Performance
        win_rate = (trade_df["return"] > 0).mean()
        avg_ret = trade_df["return"].mean()
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Avg Return: {avg_ret:.4f}")
        
        # Analysis 2: Exit Clustering
        # Estimate exit times
        trade_df["est_exit_time"] = trade_df.apply(
            lambda row: row["entry_time"] + pd.Timedelta(minutes=row["holding_mins"]) if pd.notna(row["holding_mins"]) else pd.NaT, 
            axis=1
        )
        
        # Group by Exit Time (rounded to 5 min)
        # trade_df["exit_bucket"] = trade_df["est_exit_time"].dt.round("5min")
        # exit_counts = trade_df["exit_bucket"].value_counts().sort_index()
        
        # Find heavy exit events
        # counts > 1
        # high_concurrency_exits = exit_counts[exit_counts > 1]
        
        # print(f"\nConcurrent Exit Events (>1 trade ending): {len(high_concurrency_exits)}")
        # if not high_concurrency_exits.empty:
            # print("Top 5 Concurrent Exits:")
            # print(high_concurrency_exits.sort_values(ascending=False).head(5))
            
        # Analysis 3: Stop Loss Dominance
        if "exit_reason" in trade_df.columns:
            print("\nExit Reasons:")
            print(trade_df["exit_reason"].value_counts())
            
        # Check for correlating large losses
        # When > 3 trades exit at same time with LOSS
        loss_trades = trade_df[trade_df["return"] < 0].copy()
        if not loss_trades.empty:
            loss_trades["exit_bucket"] = loss_trades["est_exit_time"].dt.round("5min")
            loss_exits = loss_trades["exit_bucket"].value_counts()
            simul_losses = loss_exits[loss_exits > 2]
            print(f"\nSimultaneous Stop Loss Events (>2 trades): {len(simul_losses)}")
            if not simul_losses.empty:
                print(simul_losses.head())

    except Exception as e:
        print(f"Error parsing report: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
        analyze_report(path)
    else:
        # Defaults
        analyze_report("c:/Users/YCL/QuantTradSys/reports/vector-1.2.1-test.html")
        print("-" * 30)
        analyze_report("c:/Users/YCL/QuantTradSys/reports/vector-1.2.1-traing.html")
