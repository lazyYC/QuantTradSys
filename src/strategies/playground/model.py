"""star_xgb 策略的模型訓練與評估。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error

from .dataset import (
    SAMPLE_WEIGHT_COLUMN,
    TARGET_COLUMN,
    apply_trade_amount_scaling,
    compute_trade_amount_stats,
    list_feature_columns,
)
from .labels import RETURN_CLASSES
from .params import StarIndicatorParams, StarModelParams

CLASS_VALUES = np.array(RETURN_CLASSES, dtype=float)
NUM_CLASSES = len(CLASS_VALUES)


@dataclass
class StarTrainingResult:
    indicator_params: StarIndicatorParams
    model_params: StarModelParams
    train_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    model_path: Path
    feature_columns: List[str]
    rankings: List[Dict[str, object]]
    class_means: List[float]
    class_thresholds: Dict[str, float]
    feature_stats: Dict[str, Dict[str, float]]


def train_star_model(
    dataset: pd.DataFrame,
    indicator_params: StarIndicatorParams,
    model_candidates: Iterable[StarModelParams],
    *,
    model_dir: Path,
    valid_dataset: Optional[pd.DataFrame] = None,
    valid_days: int = 30,
    transaction_cost: float = 0.0,
    min_validation_days: int = 30,
    stop_loss_pct: Optional[float] = None,
    use_gpu: bool = False,
    seed: Optional[int] = None,
    deterministic: bool = True,
    use_vectorized_metrics: bool = False,
) -> StarTrainingResult:
    """針對單一指標參數搜尋最佳模型。"""
    if dataset.empty:
        raise ValueError("提供的資料集為空，無法訓練模型")

    if valid_dataset is not None:
        train_df_raw = dataset.copy()
        valid_df_raw = valid_dataset.copy()
    elif valid_days <= 0:
        train_df_raw = dataset.copy()
        valid_df_raw = pd.DataFrame(columns=dataset.columns)
    else:
        validation_window = max(valid_days, min_validation_days)
        train_df_raw, valid_df_raw = _split_train_valid(
            dataset, valid_days=validation_window
        )
        if train_df_raw.empty:
            train_df_raw = dataset.copy()
            valid_df_raw = pd.DataFrame(columns=dataset.columns)

    trade_amount_stats = compute_trade_amount_stats(train_df_raw)
    prepared_dataset = apply_trade_amount_scaling(dataset, trade_amount_stats)
    dataset = prepared_dataset
    train_df = apply_trade_amount_scaling(train_df_raw, trade_amount_stats)
    valid_df = apply_trade_amount_scaling(valid_df_raw, trade_amount_stats)

    feature_cols = list_feature_columns(prepared_dataset)
    if not feature_cols:
        raise ValueError("沒有可用的特徵欄位，無法訓練模型")

    class_means = _compute_class_means(
        train_df if not train_df.empty else prepared_dataset
    )

    rankings: List[Dict[str, object]] = []
    best_tuple: (
        Tuple[StarModelParams, Dict[str, float], lgb.LGBMClassifier, np.ndarray] | None
    ) = None

    for model_params in model_candidates:
        model = _init_model(
            model_params,
            device="gpu" if use_gpu else "cpu",
            seed=seed,
            deterministic=deterministic,
        )
        fitted_model, val_probs = _fit_model(model, train_df, valid_df, feature_cols)
        val_target_df = valid_df if not valid_df.empty else train_df
        
        if use_vectorized_metrics:
            metrics = _evaluate_vectorized(
                val_target_df,
                val_probs,
                model_params,
                class_means,
                transaction_cost=transaction_cost,
            )
        else:
            metrics = _evaluate(
                val_target_df,
                val_probs,
                model_params,
                class_means,
                indicator_params=indicator_params,
                transaction_cost=transaction_cost,
                stop_loss_pct=stop_loss_pct,
                min_hold_bars=indicator_params.future_window,
            )
        score = _score_metric(metrics)
        record = {
            "indicator_params": indicator_params.as_dict(rounded=True),
            "model_params": model_params.as_dict(rounded=True),
            "metrics": metrics,
            "score": score,
        }
        rankings.append(record)
        if best_tuple is None or score > best_tuple[1]["score"]:
            record_with_score = metrics.copy()
            record_with_score["score"] = score
            best_tuple = (model_params, record_with_score, fitted_model, val_probs)

    if best_tuple is None:
        raise ValueError("模型搜尋失敗，無任何有效結果")

    best_model_params, best_metrics, _, _ = best_tuple
    rankings.sort(key=lambda item: item.get("score", 0.0), reverse=True)

    final_model = _init_model(best_model_params)
    final_model.fit(
        train_df[feature_cols],
        train_df[TARGET_COLUMN],
        sample_weight=train_df[SAMPLE_WEIGHT_COLUMN],
        eval_set=None,
    )

    # === FEATURE IMPORTANCE LOGGING ===
    try:
        # Use Gain (Total Info Gain) to identify useful vs useless features
        importances = final_model.booster_.feature_importance(importance_type='gain')
        imp_df = pd.DataFrame({
            'feature': feature_cols,
            'gain': importances
        }).sort_values(by='gain', ascending=False)

        print("\n" + "="*50)
        print("LIGHTGBM FEATURE IMPORTANCE (GAIN) - TOP 20")
        print("="*50)
        print(imp_df.head(20).to_string(index=False))
        print("-" * 50)
        print("BOTTOM 10 (POTENTIAL CANDIDATES FOR REMOVAL)")
        print(imp_df.tail(10).to_string(index=False))
        print("="*50 + "\n")
    except Exception as e:
        print(f"[WARNING] Could not print feature importance: {e}")
    # ==================================
    train_probs = (
        final_model.predict_proba(train_df[feature_cols])
        if not train_df.empty
        else np.empty((0, NUM_CLASSES))
    )
    if not train_df.empty and train_probs.size:
        train_expected = _expected_returns(train_probs, class_means)
        train_pred_classes = CLASS_VALUES[np.argmax(train_probs, axis=1)]
        train_trades = _simulate_trades(
            train_df,
            train_expected,
            train_pred_classes,
            best_model_params,
            indicator_params=indicator_params,
            transaction_cost=transaction_cost,
            stop_loss_pct=stop_loss_pct,
            min_hold_bars=indicator_params.future_window,
        )
        class_means = _calibrate_class_means(class_means, train_trades)
    train_metrics = _evaluate(
        train_df,
        train_probs,
        best_model_params,
        class_means,
        indicator_params=indicator_params,
        transaction_cost=transaction_cost,
        stop_loss_pct=stop_loss_pct,
        min_hold_bars=indicator_params.future_window,
    )
    valid_probs = (
        final_model.predict_proba(valid_df[feature_cols])
        if not valid_df.empty
        else np.empty((0, NUM_CLASSES))
    )
    test_metrics = _evaluate(
        valid_df,
        valid_probs,
        best_model_params,
        class_means,
        indicator_params=indicator_params,
        transaction_cost=transaction_cost,
        stop_loss_pct=stop_loss_pct,
        min_hold_bars=indicator_params.future_window,
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    model_filename = (
        f"star_xgb_{indicator_params.trend_window}_{best_model_params.num_leaves}.txt"
    )
    model_path = model_dir / model_filename
    final_model.booster_.save_model(str(model_path))

    threshold_source = train_df if not train_df.empty else prepared_dataset
    class_thresholds = {
        "q10": float(threshold_source["q10"].iloc[0])
        if "q10" in threshold_source.columns and not threshold_source.empty
        else 0.0,
        "q25": float(threshold_source["q25"].iloc[0])
        if "q25" in threshold_source.columns and not threshold_source.empty
        else 0.0,
        "q75": float(threshold_source["q75"].iloc[0])
        if "q75" in threshold_source.columns and not threshold_source.empty
        else 0.0,
        "q90": float(threshold_source["q90"].iloc[0])
        if "q90" in threshold_source.columns and not threshold_source.empty
        else 0.0,
    }

    result = StarTrainingResult(
        indicator_params=indicator_params,
        model_params=best_model_params,
        train_metrics=train_metrics,
        validation_metrics=best_metrics,
        test_metrics=test_metrics,
        model_path=model_path,
        feature_columns=feature_cols,
        rankings=rankings,
        class_means=class_means.tolist(),
        class_thresholds=class_thresholds,
        feature_stats={"trade_amount": trade_amount_stats},
    )
    return result


def export_rankings(rankings: Sequence[Dict[str, object]], path: Path) -> Path:
    """將搜尋排名寫入 JSON，方便報表閱讀。"""
    data = [
        {
            "indicator_params": item.get("indicator_params"),
            "model_params": item.get("model_params"),
            "metrics": item.get("metrics"),
            "score": item.get("score"),
        }
        for item in rankings
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _simulate_trades(
    df: pd.DataFrame,
    expected_returns: np.ndarray,
    pred_classes: np.ndarray,
    params: StarModelParams,
    *,
    indicator_params: Optional[StarIndicatorParams] = None,
    transaction_cost: float = 0.0,
    stop_loss_pct: Optional[float] = None,
    min_hold_bars: int = 0,
    profit_ratio: float = 0.4,
) -> pd.DataFrame:
    """執行波動突破模擬交易：ML 預測高波動 → BB 突破進場 → ATR 追蹤止損。"""
    trade_columns = [
        "side",
        "entry_time",
        "exit_time",
        "entry_price",
        "exit_price",
        "return",
        "holding_mins",
        "entry_volatility_score", # Replaces entry_expected_return
        "exit_volatility_score",  # Replaces exit_expected_return
        "entry_class",
        "exit_class",
        "exit_reason",
        "entry_zscore",
        "exit_zscore",
    ]
    if df.empty:
        return pd.DataFrame(columns=trade_columns)

    frame = df.copy()
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    
    frame["volatility_score"] = expected_returns
    frame["predicted_class_sim"] = pred_classes

    frame = frame.dropna(
        subset=["timestamp", "close", "volatility_score"]
    )

    # --- Strategy Parameters (x5 for 1min timeframe) ---
    ind_params = indicator_params if indicator_params else StarIndicatorParams(
        trend_window=1000, slope_window=100, atr_window=70, volatility_window=100,
        volume_window=100, pattern_lookback=15, upper_shadow_min=0.65,
        body_ratio_max=0.3, volume_ratio_max=0.6, future_window=60, future_return_threshold=0.01
    )

    atr_mult = getattr(ind_params, "atr_trailing_mult", 3.0)
    trigger_thresh = getattr(ind_params, "trigger_threshold", 0.6)

    # Bollinger Bands
    bb_window = getattr(ind_params, "bb_window", 100)
    bb_std = getattr(ind_params, "bb_std", 2.0)
    roll_mean = frame["close"].rolling(bb_window).mean().shift(1)
    roll_std = frame["close"].rolling(bb_window).std(ddof=0).shift(1)
    frame["bb_upper"] = roll_mean + (roll_std * bb_std)
    frame["bb_lower"] = roll_mean - (roll_std * bb_std)

    # ATR
    if "atr" not in frame.columns:
        tr = np.maximum(frame["high"] - frame["low"], np.abs(frame["high"] - frame["close"].shift(1)))
        frame["atr"] = tr.rolling(70).mean().bfill()
        
    records = []
    open_trades: List[Dict] = []
    
    # Extract Arrays for Speed
    timestamps = frame["timestamp"].values
    closes = frame["close"].values
    highs = frame["high"].values
    lows = frame["low"].values
    bb_uppers = frame["bb_upper"].values
    bb_lowers = frame["bb_lower"].values
    probs = frame["volatility_score"].values
    atrs = frame["atr"].values
    
    # --- Feature Extraction for Filters ---
    trend_ema_window = getattr(ind_params, "trend_ema_window", 200)
    adx_min = getattr(ind_params, "adx_min", 0.0)
    
    # Pre-calculate Trend EMA
    trend_ema = frame["close"].ewm(span=trend_ema_window, adjust=False).mean().values
    
    # ADX extraction
    if "adx" in frame.columns:
        adx_values = frame["adx"].values
    else:
        adx_values = np.zeros(len(closes))
        
    # Volume Force extraction
    if "volume_force" in frame.columns and "volume_force_ma" in frame.columns:
        vol_force = frame["volume_force"].values
        vol_force_ma = frame["volume_force_ma"].values
    else:
        vol_force = np.zeros(len(closes))
        vol_force_ma = np.zeros(len(closes))
    
    max_open = ind_params.max_open_trades
    
    # Cooldown tracking: prevent same-side re-entry on same bar after stop-loss
    cooldown_bar_idx = -1
    cooldown_side = None

    for i in range(len(closes)):
        current_time = timestamps[i]
        close_price = closes[i]
        high_price = highs[i]
        low_price = lows[i]
        dh = bb_uppers[i] # Alias for consistency
        dl = bb_lowers[i] # Alias for consistency
        prob = probs[i]
        atr = atrs[i]
        
        # --- 1. Manage Open Trades (Trailing Stop) ---
        new_open_trades = []
        for t in open_trades:
            side = t["side"]
            entry = t["entry_price"]
            # Get existing state
            highest_high = t.get("highest_high", entry)
            lowest_low = t.get("lowest_low", entry)
            prev_stop = t.get("stop_price", entry) 
            
            should_exit = False
            exit_fill_price = 0.0
            
            if side == "LONG":
                # Check exit condition first to ensure safety (prevent self-sabotage on big candles)
                if low_price <= prev_stop:
                    should_exit = True
                    exit_fill_price = close_price  # Exit at bar close (actual trading logic)
                else:
                    # If Safe -> Update for NEXT bar
                    if high_price > highest_high:
                        highest_high = high_price
                        t["highest_high"] = highest_high
                    
                    # Monotonic Calculation
                    raw_stop = highest_high - (atr * atr_mult)
                    new_stop = max(prev_stop, raw_stop)
                    t["stop_price"] = new_stop
                    
            else: # SHORT
                # Check Exit First
                if high_price >= prev_stop:
                    should_exit = True
                    exit_fill_price = close_price  # Exit at bar close (actual trading logic)
                else:
                    # If Safe -> Update for NEXT bar
                    if low_price < lowest_low:
                        lowest_low = low_price
                        t["lowest_low"] = lowest_low
                        
                    raw_stop = lowest_low + (atr * atr_mult)
                    new_stop = min(prev_stop, raw_stop)
                    t["stop_price"] = new_stop
            
            if should_exit:
                # Calculate PnL (assuming perfect fill at stop price)
                exit_fill = exit_fill_price
                
                ret = (exit_fill - entry)/entry if side == "LONG" else (entry - exit_fill)/entry
                gross_ret = ret
                if transaction_cost: ret -= transaction_cost
                
                # Determine exit reason
                if gross_ret > 0:
                    exit_reason = "trailing_stop_profit"
                else:
                    exit_reason = "trailing_stop_loss"
                
                delta = pd.to_datetime(current_time) - pd.to_datetime(t["entry_time"])
                
                records.append({
                    "side": side,
                    "entry_time": t["entry_time"],
                    "exit_time": current_time,
                    "entry_price": entry,
                    "exit_price": exit_fill,
                    "return": ret,
                    "holding_mins": delta.total_seconds() / 60.0,
                    "entry_volatility_score": t["entry_volatility_score"],
                    "exit_volatility_score": prob,
                    "entry_class": t["entry_class"],
                    "exit_class": 0,
                    "exit_reason": exit_reason,
                    "entry_zscore": 0.0,
                    "exit_zscore": 0.0,
                })
                # Set cooldown to prevent same-side re-entry on same bar
                cooldown_bar_idx = i
                cooldown_side = side
            else:
                new_open_trades.append(t)
        
        open_trades = new_open_trades
        
        # --- 2. Entry Logic ---
        # Single position per side logic (or max_open check)
        
        if len(open_trades) >= max_open:
            continue
            
        # Trigger Condition: AI says "Unsafe" (High Volatility Incoming)
        if prob > trigger_thresh:
            
            # --- Hard Filters Check ---
            
            # 1. ADX Filter (Avoid Choppy "Breakouts")
            if adx_values[i] < adx_min:
                continue
                
            # 2. Trend Alignment (Long only if > EMA, Short only if < EMA)
            trend_ok_long = True
            trend_ok_short = True
            if getattr(ind_params, "require_trend_alignment", False):
                trend_val = trend_ema[i]
                if close_price < trend_val:
                    trend_ok_long = False
                if close_price > trend_val:
                    trend_ok_short = False
            
            # 3. Volume Confirmation (Volume Force > Avg)
            vol_ok_long = True
            vol_ok_short = True
            if getattr(ind_params, "volume_confirmation", False):
                vf = vol_force[i]
                vf_ma = abs(vol_force_ma[i])
                if vf <= 0: vol_ok_long = False
                if vf >= 0: vol_ok_short = False

            # Action Condition: Bollinger Band Breakout
            # DH/DL are now BB Upper/Lower
            
            # Long Breakout (High touches BB Upper = Intrabar Breakout)
            if high_price > dh and not np.isnan(dh):
                if not trend_ok_long or not vol_ok_long:
                    continue
                
                # Cooldown check: skip if just stopped out from LONG on this bar
                if cooldown_bar_idx == i and cooldown_side == "LONG":
                    continue

                has_long = any(t["side"] == "LONG" for t in open_trades)
                if not has_long:
                    # Entry at close (we only know breakout happened at bar close)
                    init_stop = close_price - (atr * atr_mult)
                    open_trades.append({
                        "side": "LONG",
                        "entry_time": current_time,
                        "entry_price": close_price,
                        "entry_volatility_score": prob,
                        "entry_class": 1,
                        "highest_high": high_price,
                        "stop_price": init_stop,
                    })
            
            # Short Breakout (Low touches BB Lower = Intrabar Breakout)
            elif low_price < dl and not np.isnan(dl):
                if not trend_ok_short or not vol_ok_short:
                    continue
                
                # Cooldown check: skip if just stopped out from SHORT on this bar
                if cooldown_bar_idx == i and cooldown_side == "SHORT":
                    continue

                has_short = any(t["side"] == "SHORT" for t in open_trades)
                if not has_short:
                    # Entry at close (we only know breakout happened at bar close)
                    init_stop = close_price + (atr * atr_mult)
                    open_trades.append({
                        "side": "SHORT",
                        "entry_time": current_time,
                        "entry_price": close_price,
                        "entry_volatility_score": prob,
                        "entry_class": 1,
                        "lowest_low": low_price,
                        "stop_price": init_stop,
                    })

    # Cleanup Exit
    if open_trades:
        last_row = frame.iloc[-1]
        last_ts = last_row["timestamp"]
        last_price = float(last_row["close"])
        for t in open_trades:
            side = t["side"]
            ret = (last_price - t["entry_price"]) / t["entry_price"] if side == "LONG" else (t["entry_price"] - last_price) / t["entry_price"]
            if transaction_cost: ret -= transaction_cost
            
            records.append({
                "side": side,
                "entry_time": t["entry_time"],
                "exit_time": last_ts,
                "entry_price": t["entry_price"],
                "exit_price": last_price,
                "return": ret,
                "holding_mins": 0.0,
                "entry_volatility_score": t["entry_volatility_score"],
                "exit_volatility_score": 0.0,
                "entry_class": t["entry_class"],
                "exit_class": 0,
                "exit_reason": "end_of_data",
                "entry_zscore": t.get("entry_zscore", 0.0),
                "exit_zscore": 0.0,
            })

    return pd.DataFrame.from_records(records, columns=trade_columns)


def _summarize_trades(trades: pd.DataFrame) -> Dict[str, float]:
    """從交易表計算關鍵統計，所有報酬以幾何方式累積。"""
    summary: Dict[str, float] = {
        "trades": 0.0,
        "total_return": 0.0,
        "avg_return": 0.0,
        "win_rate": 0.0,
        "max_drawdown": 0.0,
        "short_trades": 0.0,
        "long_trades": 0.0,
        "short_total_return": 0.0,
        "long_total_return": 0.0,
        "short_win_rate": 0.0,
        "long_win_rate": 0.0,
        "mean_expected_short": 0.0,
        "mean_expected_long": 0.0,
        "short_avg_best_return": 0.0,
        "long_avg_best_return": 0.0,
    }

    if trades.empty:
        return summary

    returns = pd.to_numeric(trades["return"], errors="coerce").dropna()
    if returns.empty:
        return summary

    equity = (1.0 + returns).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    drawdown = equity / equity.cummax() - 1.0

    summary.update(
        {
            "trades": float(len(returns)),
            "total_return": total_return,
            "avg_return": float(returns.mean()),
            "win_rate": float((returns > 0).mean()),
            "max_drawdown": float(abs(drawdown.min())),
        }
    )

    for side, key in (("SHORT", "short"), ("LONG", "long")):
        mask = trades["side"] == side
        side_returns = pd.to_numeric(
            trades.loc[mask, "return"], errors="coerce"
        ).dropna()
        if not side_returns.empty:
            summary[f"{key}_trades"] = float(len(side_returns))
            summary[f"{key}_total_return"] = float(np.prod(1.0 + side_returns) - 1.0)
            summary[f"{key}_win_rate"] = float((side_returns > 0).mean())
            
            # Use new column name, fallback to old if missing
            col_name = "entry_volatility_score" if "entry_volatility_score" in trades.columns else "entry_expected_return"
            expected_series = pd.to_numeric(
                trades.loc[mask, col_name], errors="coerce"
            ).dropna()

            summary[f"mean_volatility_score_{key}"] = (
                float(expected_series.mean()) if not expected_series.empty else 0.0
            )
            summary[f"mean_expected_{key}"] = summary[f"mean_volatility_score_{key}"]
        else:
            summary[f"{key}_trades"] = 0.0
            summary[f"{key}_total_return"] = 0.0
            summary[f"{key}_win_rate"] = 0.0
            summary[f"mean_volatility_score_{key}"] = 0.0
            summary[f"mean_expected_{key}"] = 0.0

    return summary


def _split_train_valid(
    train_df: pd.DataFrame, valid_days: int = 30
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if train_df.empty:
        return train_df.copy(), train_df.copy()
    cutoff = train_df["timestamp"].max() - pd.Timedelta(days=valid_days)
    valid = train_df[train_df["timestamp"] > cutoff].copy()
    base = train_df[train_df["timestamp"] <= cutoff].copy()
    if valid.empty or base.empty:
        split_idx = max(int(len(train_df) * 0.8), 1)
        base = train_df.iloc[:split_idx].copy()
        valid = train_df.iloc[split_idx:].copy()
    return base.reset_index(drop=True), valid.reset_index(drop=True)


def _init_model(
    params: StarModelParams,
    *,
    device: str = "cpu",
    seed: Optional[int] = None,
    deterministic: bool = True,
) -> lgb.LGBMClassifier:
    seed_val = 88 if seed is None else seed
    return lgb.LGBMClassifier(
        objective="multiclass",
        boosting_type="gbdt",
        num_leaves=params.num_leaves,
        max_depth=params.max_depth,
        learning_rate=params.learning_rate,
        n_estimators=params.n_estimators,
        min_child_samples=params.min_child_samples,
        subsample=params.subsample,
        colsample_bytree=params.colsample_bytree,
        feature_fraction_bynode=params.feature_fraction_bynode,
        reg_alpha=params.lambda_l1,
        reg_lambda=params.lambda_l2,
        bagging_freq=params.bagging_freq,
        num_class=NUM_CLASSES,
        n_jobs=-1,
        verbosity=-1,
        random_state=seed_val,
        bagging_seed=seed_val,
        feature_fraction_seed=seed_val,
        deterministic=deterministic,
        device=device,
    )


def _fit_model(
    model: lgb.LGBMClassifier,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> Tuple[lgb.LGBMClassifier, np.ndarray]:
    eval_set = None
    eval_sample_weight = None
    callbacks = []
    if not valid_df.empty:
        eval_set = [(valid_df[feature_cols], valid_df[TARGET_COLUMN])]
        eval_sample_weight = [valid_df[SAMPLE_WEIGHT_COLUMN]]
        callbacks.append(lgb.early_stopping(stopping_rounds=50, verbose=False))
    model.fit(
        train_df[feature_cols],
        train_df[TARGET_COLUMN],
        sample_weight=train_df[SAMPLE_WEIGHT_COLUMN],
        eval_set=eval_set,
        eval_sample_weight=eval_sample_weight,
        eval_metric="multi_logloss",
        callbacks=callbacks,
    )
    target_df = valid_df if not valid_df.empty else train_df
    probs = model.predict_proba(target_df[feature_cols])
    return model, probs


def _expected_returns(probs: np.ndarray, class_means: np.ndarray) -> np.ndarray:
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)
    return probs @ class_means


def _evaluate(
    df: pd.DataFrame,
    probs: np.ndarray,
    params: StarModelParams,
    class_means: np.ndarray,
    *,
    indicator_params: Optional[StarIndicatorParams] = None,
    transaction_cost: float = 0.0,
    stop_loss_pct: Optional[float] = None,
    min_hold_bars: int = 0,
) -> Dict[str, float]:
    if df.empty:
        metrics = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "threshold": float(params.decision_threshold),
        }
        metrics.update(_summarize_trades(pd.DataFrame()))
        metrics["score"] = 0.0
        return metrics

    # Probs is [N, 2] usually. We want Prob(Unsafe) = Prob(1)
    # If using LightGBM multiclass=2, it returns [Prob0, Prob1].
    volatility_score = probs[:, 1]
    
    # Predict based on Decision Threshold
    preds = (volatility_score > params.decision_threshold).astype(int)
    y_true = df[TARGET_COLUMN].to_numpy()
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    accuracy = float(accuracy_score(y_true, preds))
    
    # Binary Metrics (Positive Class = Unsafe)
    # True Positive = Predicted Unsafe AND Is Unsafe (Good detection)
    # False Negative = Predicted Safe BUT Is Unsafe (Missed Trend -> Disaster)
    # False Positive = Predicted Unsafe BUT Is Safe (False Alarm -> Opportunity Cost)
    
    precision = float(precision_score(y_true, preds, zero_division=0))
    recall = float(recall_score(y_true, preds, zero_division=0))
    
    # Simulation
    trades = _simulate_trades(
        df,
        volatility_score,
        preds,
        params,
        indicator_params=indicator_params,
        transaction_cost=transaction_cost,
        stop_loss_pct=stop_loss_pct,
        min_hold_bars=min_hold_bars,
    )
    metric_summary = _summarize_trades(trades)
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall, 
        "threshold": float(params.decision_threshold),
    }
    metrics.update(metric_summary)
    
    # Score Calculation: Focus on Avoiding Disaster (Min Drawdown) + Profit
    # High Recall is good (Caught all unsafe moments).
    # But ultimate test is PnL / Drawdown.
    
    total_return = metric_summary["total_return"]
    max_dd = metric_summary["max_drawdown"]
    
    # Score: Calmar Ratio-ish
    if max_dd > 0:
        score = total_return / (max_dd * 2.0) # Penalize DD more
    else:
        score = total_return
        
    metrics["score"] = score
    return metrics


def _evaluate_vectorized(
    df: pd.DataFrame,
    probs: np.ndarray,
    params: StarModelParams,
    class_means: np.ndarray,
    *,
    transaction_cost: float = 0.0,
) -> Dict[str, float]:
    """
    Vectorized evaluation for random splits where path dependency is broken.
    Uses pre-calculated future_long_return / future_short_return.
    """
    if df.empty:
        return {"score": 0.0, "total_return": 0.0, "trades": 0.0}

    # [FIXED V1.6.6] Safety Filter Vectorized Logic
    # The model predicts prob_unsafe (Class 1).
    # If prob_unsafe < threshold -> SAFE -> We assume we hold the asset (Long).
    # If prob_unsafe >= threshold -> UNSAFE -> We go to Cash (0 return).
    # This rewards the model for avoiding large drawdowns (when future_long_return is negative)
    # and punishes it for missing rallies (when future_long_return is positive).
    
    volatility_score = probs[:, 1]
    threshold = float(params.decision_threshold)
    
    # Logic: Safe vs Unsafe
    # Note: We use 'future_long_return' as the proxy for holding the asset.
    # Ideally we'd know the trend, but for random split validation, Long Bias Proxy is standard for Crypto.
    
    is_safe = volatility_score < threshold
    
    pnl = np.zeros(len(df))
    
    # If Safe, we take the market return (minus transaction cost if we were re-entering, but let's ignore freq trading here)
    # Strictly for Optuna scoring:
    pnl[is_safe] = df.loc[is_safe, "future_long_return"].fillna(0.0)
    
    # If Unsafe, pnl is 0.0 (Cash)
    
    # Total Return
    total_return = np.prod(1.0 + pnl) - 1.0
    
    trades_count = len(df) # N/A really
    win_rate = (pnl > 0).mean()
    
    return {
        "score": float(total_return),
        "total_return": float(total_return),
        "trades": float(trades_count),
        "win_rate": float(win_rate),
        "avg_return": float(pnl.mean()),
        "threshold": threshold,
    }


def _score_metric(metrics: Dict[str, float]) -> float:
    return float(metrics.get("total_return", 0.0))


def _compute_class_means(train_df: pd.DataFrame) -> np.ndarray:
    """依據多空類別計算對應的期望報酬均值。"""
    short_means = train_df.groupby(TARGET_COLUMN)["future_short_return"].mean()
    long_means = train_df.groupby(TARGET_COLUMN)["future_long_return"].mean()
    class_means: list[float] = []
    for label in CLASS_VALUES:
        if label < 0:
            value = short_means.get(label, 0.0)
        elif label > 0:
            value = long_means.get(label, 0.0)
        else:
            value = 0.0
        class_means.append(float(0.0 if pd.isna(value) else value))
    return np.asarray(class_means, dtype=float)

def _calibrate_class_means(
    base_means: np.ndarray,
    trades: pd.DataFrame,
    *,
    smoothing: float = 0.2,
) -> np.ndarray:
    """利用模擬交易的平均報酬，平滑調整 class_means 但保留原本量級。"""
    if (
        trades.empty
        or "entry_class" not in trades
        or "return" not in trades
        or smoothing <= 0.0
    ):
        return base_means

    calibrated = base_means.astype(float, copy=True)
    returns = pd.to_numeric(trades["return"], errors="coerce")
    classes = pd.to_numeric(trades["entry_class"], errors="coerce")

    smooth = min(max(smoothing, 0.0), 1.0)
    for idx, label in enumerate(CLASS_VALUES):
        mask = classes == label
        if not mask.any():
            continue
        mean_return = returns[mask].mean()
        if pd.isna(mean_return):
            continue

        base_value = calibrated[idx]
        # 只在方向一致時進行平滑，避免意外顛倒多空判斷
        if base_value == 0.0 or np.sign(base_value) == np.sign(mean_return):
            calibrated[idx] = float((1.0 - smooth) * base_value + smooth * mean_return)

    return calibrated

__all__ = [
    "StarTrainingResult",
    "train_star_model",
    "export_rankings",
    "CLASS_VALUES",
    "_simulate_trades",
    "_summarize_trades",
]
