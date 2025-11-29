import sqlite3
import sys
import json

# 1. Setup (Path, Logging, Config)
import _setup
from _setup import DEFAULT_STATE_DB

def main():
    try:
        print(f"Checking DB at: {DEFAULT_STATE_DB}")
        conn = sqlite3.connect(DEFAULT_STATE_DB)
        
        # Check schema
        cursor = conn.execute("PRAGMA table_info(strategy_params)")
        columns = [row[1] for row in cursor.fetchall()]
        print(f"Columns: {columns}")
        
        # Check strategy_trades
        cursor = conn.execute("SELECT count(*) FROM strategy_trades")
        count = cursor.fetchone()[0]
        print(f"Trades found: {count}")

        # Check content
        cursor = conn.execute("SELECT strategy, study, symbol, timeframe FROM strategy_params")
        rows = cursor.fetchall()
        print(f"Rows found: {len(rows)}")
        for row in rows:
            print(f"Row: {row}")
            
        # Check specific query (Example)
        cursor = conn.execute(
            "SELECT params_json, model_path FROM strategy_params WHERE strategy='star_xgb' AND study='test3' ORDER BY updated_at DESC LIMIT 1"
        )
        row = cursor.fetchone()
        if row:
            print("Record FOUND for test3")
            params = json.loads(row[0])
            print(f"Params keys: {list(params.keys())}")
            print(f"Model Path in DB column: {row[1]}")
            print(f"Model Path in params_json: {params.get('model_path')}")
            print(f"Feature Columns in params: {'feature_columns' in params}")
            print(f"Class Means in params: {'class_means' in params}")
            print(f"Feature Stats in params: {'feature_stats' in params}")
        else:
            print("Record NOT FOUND for test3")
            
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
