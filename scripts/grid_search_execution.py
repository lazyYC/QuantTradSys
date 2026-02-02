
import logging
import sys
import pandas as pd
import numpy as np
import dataclasses
import itertools
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

# Setup paths to src
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from persistence.param_store import load_strategy_params
from data_pipeline.ccxt_fetcher import fetch_yearly_ohlcv
from utils.data_utils import prepare_ohlcv_frame
from strategies.playground.adapter import PlaygroundStrategy
from strategies.playground.backtest import load_star_model, _simulate_trades
from strategies.playground.features import StarFeatureCache
from strategies.playground.params import StarIndicatorParams, StarModelParams
from strategies.playground.dataset import prepend_warmup_rows
from strategies.playground.model import StarTrainingResult # Keep other imports if needed, or remove line if empty

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)

def split_train_valid_test(df: pd.DataFrame, test_days: int = 30, valid_pct: float = 0.150) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split logic matching optimization v1.8.4 incentive alignment.
    1. Test = Last `test_days`
    2. Valid = Last `valid_pct` of the remaining
    3. Train = The rest
    """
    if df.empty:
        return df, df, df

    # 1. Cut Test (Tail)
    test_cutoff = df["timestamp"].max() - pd.Timedelta(days=test_days)
    test_df = df[df["timestamp"] > test_cutoff].copy()
    remaining_df = df[df["timestamp"] <= test_cutoff].copy()
    
    if remaining_df.empty:
        # Fallback if dataset too small
        return remaining_df, pd.DataFrame(), test_df

    # 2. Cut Valid (Tail of Remaining)
    # Time-Series Split: Valid is the "future" relative to Train
    valid_size = int(len(remaining_df) * valid_pct)
    if valid_size < 1:
        # Fallback
        return remaining_df, pd.DataFrame(), test_df
        
    valid_df = remaining_df.iloc[-valid_size:].copy()
    train_df = remaining_df.iloc[:-valid_size].copy()
    
    return train_df, valid_df, test_df

def run_grid_search(
    strategy_name: str = "playground", 
    study_name: str = "v1.8.4_real_opt", 
    output_path: str = "grid_search_results.csv"
):
    LOGGER.info(f"Starting Grid Search for {strategy_name}/{study_name}")
    
    # 1. Load Model & Params
    record = load_strategy_params(strategy_name, study_name)
    if not record:
        LOGGER.error(f"No parameters found for {strategy_name}/{study_name}")
        return

    LOGGER.info(f"Loaded Best Params from {record.updated_at}")
    LOGGER.info(f"Model Path: {record.model_path}")
    
    symbol = record.symbol
    timeframe = record.timeframe
    
    # Reconstruct Objects
    if "indicator" in record.params:
        ind_dict = record.params["indicator"]
        mod_dict = record.params["model"]
    else:
        # Fallback for flat structure
        LOGGER.warning("Params 'indicator' key missing, assuming flat structure.")
        LOGGER.info(f"Available Keys: {list(record.params.keys())}")
        ind_dict = record.params
        mod_dict = record.params

    # Helper to filter dataclass fields
    def filter_params(cls, data):
        valid = set(cls.__annotations__.keys())
        return {k: v for k, v in data.items() if k in valid}

    indicator_params = StarIndicatorParams(**filter_params(StarIndicatorParams, ind_dict))
    model_params = StarModelParams(**filter_params(StarModelParams, mod_dict))
    
    # 2. Load Data
    raw_df = fetch_yearly_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        lookback_days=360,
        exchange_id="binanceusdm",
    )
    cleaned = prepare_ohlcv_frame(raw_df, timeframe)
    
    # 3. Feature Generation (Expensive, do once)
    LOGGER.info("Building Features...")
    cache = StarFeatureCache(
        cleaned,
        trend_windows=[indicator_params.trend_window],
        atr_windows=[indicator_params.atr_window],
        volatility_windows=[indicator_params.volatility_window],
        volume_windows=[indicator_params.volume_window],
        pattern_windows=[indicator_params.pattern_lookback],
    )
    features = cache.build_features(indicator_params)
    
    # 4. Predict Probabilities (Expensive, do once)
    LOGGER.info("Running Model Inference...")
    booster = load_star_model(record.model_path)
    
    # Get feature columns from stored params if available, else standard list
    feature_cols = record.params.get("feature_columns")
    if not feature_cols:
        # Fallback to feature df columns excluding meta
        feature_cols = [c for c in features.columns if c not in ["timestamp", "open", "high", "low", "close", "volume"]]
    
    # Feature Columns: Ask the model what it wants!
    model_expected_features = booster.feature_name()
    if model_expected_features:
        feature_cols = model_expected_features
        # Verify all are present
        missing = [c for c in feature_cols if c not in features.columns]
        if missing:
            LOGGER.warning(f"Missing features required by model: {missing}")
            LOGGER.warning("Filling missing features with 0.0 to allow inference (Likely targets used in training).")
            for c in missing:
                features[c] = 0.0
    else:
        # Fallback if no feature names stored
        exclude_targets = ["future_return", "max_upside", "max_downside", "future_long_return", "future_short_return"]
        feature_cols = [c for c in feature_cols if c not in exclude_targets and c in features.columns]

    LOGGER.info(f"Using {len(feature_cols)} features for inference.")
    
    X = features[feature_cols].to_numpy(dtype=float, copy=False)
    # Disable shape check mainly if we suspect minor metadata mismatch, but size must match
    probs = booster.predict(X, predict_disable_shape_check=False) # Should match now
    
    if probs.ndim == 1:
        volatility_score = probs
    else:
        volatility_score = probs[:, 1]

    # Debug: Volatility Score Calibration Check
    p_mean = np.mean(volatility_score)
    p_max = np.max(volatility_score)
    p_90 = np.percentile(volatility_score, 90)
    p_99 = np.percentile(volatility_score, 99)
    LOGGER.info(f"Volatility Score Stats: Mean={p_mean:.4f}, Max={p_max:.4f}, p90={p_90:.4f}, p99={p_99:.4f}")
    if p_max < 0.5:
        LOGGER.warning("Max Volatility Score < 0.5. Trigger threshold 0.55+ will NEVER fire!")
    
    full_df = features.copy()
    full_df["volatility_score"] = volatility_score
    
    # 5. Split Dataset
    train_raw, valid_raw, test_raw = split_train_valid_test(full_df, test_days=30, valid_pct=0.15)
    
    # 6. Apply Warmup for simulation rolling windows
    warmup_bars = 200

    def get_split_with_warmup(split_df):
        if split_df.empty: return split_df
        start_idx = split_df.index[0]
        warmup_start = max(0, start_idx - warmup_bars)
        # Use full_df slice
        return full_df.iloc[warmup_start : split_df.index[-1] + 1].copy(), split_df["timestamp"].min()

    train_input, train_start = get_split_with_warmup(train_raw)
    valid_input, valid_start = get_split_with_warmup(valid_raw)
    test_input, test_start = get_split_with_warmup(test_raw)
    
    LOGGER.info(f"Splits Prepared: Train={len(train_input)} (Core={len(train_raw)}), Valid={len(valid_input)} (Core={len(valid_raw)}), Test={len(test_input)} (Core={len(test_raw)})")
    
    # Define Grid
    triggers = [0.6, 0.65, 0.70, 0.75, 0.8]
    bb_stds = [1.8, 2.0, 2.2]
    
    atr_mults = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    
    # New Filter Params
    adx_mins = [20.0, 25.0, 30.0, 40.0]
    trend_aligns = [True] # Force trend alignment for safety in chop
    
    grid = list(itertools.product(triggers, bb_stds, atr_mults, adx_mins, trend_aligns))
    results = []
    
    LOGGER.info(f"Running Grid Search (BB Breakout) on {len(grid)} combinations...")
    
    for (trig, bb_dev, atr_m, adx_m, t_align) in tqdm(grid):
        # We run simulation for each split
        
        # Override params object using replace
        current_ind_params = dataclasses.replace(
            indicator_params,
            bb_std=bb_dev, # New
            bb_window=20,  # Fixed
            atr_trailing_mult=atr_m,
            trigger_threshold=trig,
            adx_min=adx_m,
            require_trend_alignment=t_align
        )
        
        row = {
            "trigger_threshold": trig,
            "bb_std": bb_dev,
            "atr_trailing_mult": atr_m,
            "adx_min": adx_m,
            "require_trend_alignment": t_align,
        }
        
        for split_name, split_df, core_start in [
            ("train", train_input, train_start),
            ("valid", valid_input, valid_start),
            ("test", test_input, test_start)
        ]:
            if split_df.empty:
                continue
                
            trades = _simulate_trades(
                split_df,
                split_df["volatility_score"].values,
                np.zeros(len(split_df)), # pred_classes
                model_params, # params (StarModelParams)
                indicator_params=current_ind_params,
                transaction_cost=0.001,
                stop_loss_pct=None, 
            )
            
            # Filter trades by actual core period
            core_trades = []
            if not trades.empty:
                cutoff_ts = pd.to_datetime(core_start, utc=True)
                # Iterate rows
                for _, t in trades.iterrows():
                    if pd.to_datetime(t["entry_time"], utc=True) >= cutoff_ts:
                        core_trades.append(t.to_dict())
            
            # evaluate
            metrics = _evaluate_simulation_from_trades(core_trades)
            
            # Flatten metrics into row
            for k, v in metrics.items():
                row[f"{split_name}_{k}"] = v
        
        results.append(row)
        
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_path, index=False)
    LOGGER.info(f"Grid Search Complete. Results saved to {output_path}")

def _evaluate_simulation_from_trades(trades: List[Dict]) -> Dict[str, float]:
    """Micro evaluation version of backtest._evaluate"""
    if not trades:
        return {
            "total_return": 0.0,
            "trades": 0,
            "win_rate": 0.0,
            "avg_holding": 0.0,
            "sharpe": 0.0
        }
        
    df = pd.DataFrame(trades)
    returns = pd.to_numeric(df["return"])
    
    total_return = np.prod(1.0 + returns) - 1.0
    win_rate = (returns > 0).mean()
    avg_holding = df["holding_mins"].mean()
    
    # Sharpe Approximation (per trade not daily, rough proxy)
    std = returns.std()
    sharpe = (returns.mean() / std * np.sqrt(len(trades))) if std > 1e-9 else 0.0
    
    return {
        "total_return": total_return,
        "trades": len(trades),
        "win_rate": win_rate,
        "avg_holding": avg_holding,
        "sharpe": sharpe
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="playground")
    parser.add_argument("--study", default="v1.8.4_real_opt") # Default to user's likely study
    parser.add_argument("--output", default="grid_search_results.csv")
    args = parser.parse_args()
    
    run_grid_search(args.strategy, args.study, args.output)
