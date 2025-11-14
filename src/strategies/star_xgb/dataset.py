"""star_xgb 策略資料集建構模組。"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


TARGET_COLUMN = "return_class"
SAMPLE_WEIGHT_COLUMN = "sample_weight"
DEFAULT_WARMUP_BARS = 60


def build_training_dataset(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    class_thresholds: Dict[str, float],
    min_abs_future_return: Optional[float] = None,
) -> pd.DataFrame:
    """合併特徵與標籤，產生訓練用的完整資料表。"""
    df = features.merge(labels, on="timestamp", how="inner", suffixes=("", "_label"))
    df = df.dropna(subset=["future_short_return", "future_long_return"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df[SAMPLE_WEIGHT_COLUMN] = 1.0
    if min_abs_future_return and min_abs_future_return > 0:
        mask = df["future_long_return"].abs() >= float(min_abs_future_return)
        df = df[mask].reset_index(drop=True)
    df["q10"] = class_thresholds.get("q10", 0.0)
    df["q25"] = class_thresholds.get("q25", 0.0)
    df["q75"] = class_thresholds.get("q75", 0.0)
    df["q90"] = class_thresholds.get("q90", 0.0)
    return df


def split_train_test(
    dataset: pd.DataFrame,
    *,
    test_days: int = 30,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """以時間序切分資料成訓練與測試集。"""
    if dataset.empty:
        return dataset.copy(), dataset.copy()
    cutoff = dataset["timestamp"].max() - pd.Timedelta(days=test_days)
    test = dataset[dataset["timestamp"] > cutoff].copy()
    train = dataset[dataset["timestamp"] <= cutoff].copy()
    if train.empty:
        cutoff = dataset["timestamp"].min() + pd.Timedelta(days=test_days)
        train = dataset[dataset["timestamp"] < cutoff].copy()
        test = dataset[dataset["timestamp"] >= cutoff].copy()
    return train.reset_index(drop=True), test.reset_index(drop=True)


def compute_trade_amount_stats(df: pd.DataFrame) -> Dict[str, float]:
    """計算 trade_amount 欄位的平均與標準差。"""
    if df.empty or "trade_amount" not in df.columns:
        return {"mean": 0.0, "std": 1.0}
    series = pd.to_numeric(df["trade_amount"], errors="coerce").dropna()
    if series.empty:
        return {"mean": 0.0, "std": 1.0}
    mean = float(series.mean())
    std = float(series.std(ddof=0))
    if not np.isfinite(std) or std == 0.0:
        std = 1.0
    return {"mean": mean, "std": std}


def apply_trade_amount_scaling(
    df: pd.DataFrame, stats: Optional[Dict[str, float]]
) -> pd.DataFrame:
    """根據既有統計量，將 trade_amount 轉換成 z-score。"""
    if df.empty:
        return df.copy()
    scaled = df.copy()
    if "trade_amount" not in scaled.columns or not stats:
        return scaled
    mean = float(stats.get("mean", 0.0))
    std = float(stats.get("std", 1.0)) or 1.0
    scaled["trade_amount_zscore"] = (scaled["trade_amount"] - mean) / std
    scaled = scaled.drop(columns=["trade_amount"], errors="ignore")
    return scaled


def prepend_warmup_rows(
    full_df: pd.DataFrame,
    target_df: pd.DataFrame,
    warmup_bars: int = DEFAULT_WARMUP_BARS,
) -> pd.DataFrame:
    """在切分後的資料前加入 warm-up K 線，避免指標初值偏差。"""
    if warmup_bars <= 0 or target_df.empty or full_df.empty:
        return target_df.copy()

    core_start = target_df["timestamp"].iloc[0]
    mask = full_df["timestamp"] >= core_start
    if not mask.any():
        return target_df.copy()
    first_idx = int(mask.idxmax())
    start_idx = max(first_idx - warmup_bars, 0)
    warmup_slice = full_df.iloc[start_idx:first_idx]
    if warmup_slice.empty:
        return target_df.copy()
    return pd.concat([warmup_slice, target_df], ignore_index=True)


def list_feature_columns(dataset: pd.DataFrame) -> List[str]:
    """列出可用於建模的特徵欄位（排除 meta 欄位）。"""
    excluded = {
        "timestamp",
        TARGET_COLUMN,
        SAMPLE_WEIGHT_COLUMN,
        "future_close_return",
        "future_min_return",
        "future_short_return",
        "future_best_short_return",
        "future_long_return",
        "future_best_long_return",
        "q10",
        "q25",
        "q75",
        "q90",
        "trade_amount",
        "open",
        "high",
        "low",
        "close",
    }
    return [col for col in dataset.columns if col not in excluded]


__all__ = [
    "TARGET_COLUMN",
    "SAMPLE_WEIGHT_COLUMN",
    "build_training_dataset",
    "split_train_test",
    "compute_trade_amount_stats",
    "apply_trade_amount_scaling",
    "prepend_warmup_rows",
    "DEFAULT_WARMUP_BARS",
    "list_feature_columns",
]
