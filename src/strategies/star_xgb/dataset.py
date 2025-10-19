"""star_xgb 策略的資料集建構工具。"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd


TARGET_COLUMN = "return_class"
SAMPLE_WEIGHT_COLUMN = "sample_weight"


def build_training_dataset(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    class_thresholds: Dict[str, float],
) -> pd.DataFrame:
    """合併特徵與標籤，回傳排序後的訓練資料表。"""
    df = features.merge(labels, on="timestamp", how="inner", suffixes=("", "_label"))
    df = df.dropna(subset=["future_short_return", "future_long_return"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df[SAMPLE_WEIGHT_COLUMN] = 1.0
    df["q10"] = class_thresholds.get("q10", 0.0)
    df["q25"] = class_thresholds.get("q25", 0.0)
    df["q75"] = class_thresholds.get("q75", 0.0)
    df["q90"] = class_thresholds.get("q90", 0.0)
    # select some columns and print them
    # df2 = df[["close", "future_short_return", "future_long_return", "return_class", "candidate", "q10", "q25", "q75", "q90"]]
    # print(df2.head(50))
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
    }
    return [col for col in dataset.columns if col not in excluded]


__all__ = [
    "TARGET_COLUMN",
    "SAMPLE_WEIGHT_COLUMN",
    "build_training_dataset",
    "split_train_test",
    "list_feature_columns",
]
