"""star_xgb ��������ƶ��غc�u��C"""

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
) -> pd.DataFrame:
    """�X�֯S�x�P���ҡA�^�ǱƧǫ᪺�V�m��ƪ��C"""
    df = features.merge(labels, on="timestamp", how="inner", suffixes=("", "_label"))
    df = df.dropna(subset=["future_short_return", "future_long_return"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df[SAMPLE_WEIGHT_COLUMN] = 1.0
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


def list_feature_columns(dataset: pd.DataFrame) -> List[str]:
    """列出可用於建模的特徵欄位（排除 meta 欄位）。"""
    excluded = {
        "timestamp",
        TARGET_COLUMN,
        SAMPLE_WEIGHT_COLUMN,
        "candidate",
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
    "list_feature_columns",
]
