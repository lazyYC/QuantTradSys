
import argparse
import logging
import sys
import uvicorn
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import pandas as pd
import numpy as np

# Setup paths
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from strategies.loader import load_strategy_class
from persistence.market_store import MarketDataStore
from persistence.param_store import load_strategy_params
from strategies.base import BaseStrategy
from reporting.chart_utils import format_candles, format_volume, format_equity, format_signals, format_markers
from reporting.table_html_utils import generate_metrics_html, generate_distribution_html, generate_analysis_html, generate_trades_html

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("Playground")

# Global State
SIM_STATE: Dict[str, Any] = {
    "strategy": None,
    "candles": pd.DataFrame(),
    "model_path": None,
    "base_params": {},
    "feature_cache": None,
}

class SimParams(BaseModel):
    trigger_threshold: float
    bb_std: float
    atr_trailing_mult: float
    adx_min: float
    require_trend_alignment: bool

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load Data
    args = parse_args()
    LOGGER.info(f"Loading Strategy: {args.strategy}")
    
    # 1. Load Strategy Class
    try:
        dist_cls = load_strategy_class(args.strategy)
        SIM_STATE["strategy"] = dist_cls()
    except Exception as e:
        LOGGER.error(f"Failed to load strategy: {e}")
        sys.exit(1)
        
    # 2. Load Metadata & Params
    record = load_strategy_params(args.strategy, args.study)
    if not record:
        LOGGER.error("No params found. Please train first.")
        sys.exit(1)
        
    SIM_STATE["base_params"] = record.params
    SIM_STATE["model_path"] = record.model_path
    
    # 3. Load Market Data (Last N Days)
    store = MarketDataStore()
    
    # 3.1 Sync Recent Data (ensure freshness)
    try:
        from data_pipeline.ccxt_fetcher import fetch_yearly_ohlcv
        LOGGER.info(f"Syncing market data for {record.symbol} (buffer {args.days + 5} days)...")
        fetch_yearly_ohlcv(
            symbol=record.symbol,
            timeframe=record.timeframe,
            lookback_days=args.days + 5,
            exchange_id="binanceusdm", # Default to binanceusdm as in report.py
            market_store=store
        )
    except Exception as e:
        LOGGER.warning(f"Data sync failed (proceeding with DB data): {e}")

    LOGGER.info(f"Loading Data for {record.symbol} (Last {args.days} days)...")
    
    # Resolve timestamps with efficient query
    # 1. Get latest timestamp first
    latest_ms = store.get_latest_timestamp(record.symbol, record.timeframe)
    if not latest_ms:
        LOGGER.error("No data found for symbol.")
        sys.exit(1)
        
    latest_dt = pd.to_datetime(latest_ms, unit="ms", utc=True)
    
    # 2. Calculate needed start time (Days + Warmup buffer)
    # Warmup 20 days to be safe for 200 EMA
    start_dt = latest_dt - pd.Timedelta(days=args.days + 20)
    
    LOGGER.info(f"Fetching data from {start_dt} to {latest_dt}...")
    
    # 3. Load only the slice
    df = store.load_candles(
        symbol=record.symbol,
        timeframe=record.timeframe,
        start_ts=start_dt
    )
    
    if df.empty:
        LOGGER.error("No data found in range.")
        sys.exit(1)
        
    SIM_STATE["candles"] = df.reset_index(drop=True)
    LOGGER.info(f"Loaded {len(SIM_STATE['candles'])} candles.")
    
    # 4. Pre-calc Features
    # Explicitly warm up the strategy feature cache
    LOGGER.info("Warming up feature cache...")
    SIM_STATE["strategy"].warm_up(SIM_STATE["candles"])
    LOGGER.info("Warmup complete.")
    
    yield
    # Shutdown logic if any

app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

@app.get("/", response_class=HTMLResponse)
async def get_ui(request: Request):
    return templates.TemplateResponse("playground.html", {
        "request": request,
        "defaults": {
            "trigger_threshold": 0.6, # Default to match report.py
            "bb_std": 2.2,
            "atr_trailing_mult": 3.0,
            "adx_min": 20,
        }
    })

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(content=b"", media_type="image/x-icon")

@app.get("/.well-known/appspecific/com.chrome.devtools.json", include_in_schema=False)
async def chrome_devtools():
    return Response(status_code=200, content="{}")

@app.post("/simulate")
async def run_simulation(params: SimParams):
    if SIM_STATE["candles"].empty:
        return {"error": "No data loaded"}
        
    strategy: BaseStrategy = SIM_STATE["strategy"]
    
    # Merge Request Params with Base Params
    # We copy base params and override
    current_params = SIM_STATE["base_params"].copy()
    
    # Override keys
    current_params["trigger_threshold"] = params.trigger_threshold
    current_params["bb_std"] = params.bb_std
    current_params["atr_trailing_mult"] = params.atr_trailing_mult
    current_params["adx_min"] = params.adx_min
    current_params["require_trend_alignment"] = params.require_trend_alignment
    
    # Run Backtest
    max_ts = SIM_STATE["candles"]["timestamp"].max()
    args = parse_args()
    core_start = max_ts - pd.Timedelta(days=args.days)
    
    try:
        result = strategy.backtest(
            SIM_STATE["candles"],
            current_params,
            model_path=SIM_STATE["model_path"],
            core_start=core_start
        )
    except Exception as e:
        LOGGER.error(f"Backtest Error: {e}")
        return {"error": str(e)}
        
    # Format Results for UI using shared Utils
    
    # Trim warmup from visualization
    vis_start = core_start - pd.Timedelta(days=2)
    vis_candles = SIM_STATE["candles"][SIM_STATE["candles"]["timestamp"] >= vis_start].copy()
    
    candle_data = format_candles(vis_candles)
    volume_data = format_volume(vis_candles)
    
    equity_data = format_equity(result.equity_curve)
    
    # Check signals
    prob_data = []
    if hasattr(result, "signals") and not result.signals.empty:
        sig_col = "volatility_score" if "volatility_score" in result.signals.columns else "prob_unsafe"
        prob_data = format_signals(result.signals, sig_col)
        
    markers = format_markers(result.trades)
    
    # 2. Tables (HTML)
    metrics_df = pd.DataFrame([result.metrics])
    metrics_html = generate_metrics_html(metrics_df)
    
    # Trades Tables
    dist_html = generate_distribution_html(result.trades)
    analysis_html = generate_analysis_html(result.trades)
    trades_html = generate_trades_html(result.trades)

    return {
        "metrics": result.metrics, # Keep raw for top bar if needed
        "metrics_html": metrics_html,
        "dist_html": dist_html,
        "analysis_html": analysis_html,
        "trades_html": trades_html,
        "candles": candle_data,
        "volume": volume_data,
        "equity": equity_data,
        "signals": prob_data,
        "markers": markers,
        "trade_count": len(result.trades)
    }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="playground")
    parser.add_argument("--study", default="v2.2.1")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--port", type=int, default=8000)
    # Handle unknown args (uvicorn passes some?)
    args, _ = parser.parse_known_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    uvicorn.run("interactive_backtest:app", host="0.0.0.0", port=args.port, reload=True)
