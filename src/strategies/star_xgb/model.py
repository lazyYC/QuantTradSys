
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


def _simulate_trades(
    df: pd.DataFrame,
    expected_returns: np.ndarray,
    pred_classes: np.ndarray,
    params: StarModelParams,
) -> pd.DataFrame:
    """依據預測類別與預期報酬模擬交易，回傳完整交易表。"""
    trade_columns = [
        "side",
        "entry_time",
        "exit_time",
        "entry_price",
        "exit_price",
        "return",
        "holding_mins",
        "entry_expected_return",
        "exit_expected_return",
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
    frame["expected_return_sim"] = expected_returns
    frame["predicted_class_sim"] = pred_classes
    frame = frame.dropna(subset=["timestamp", "close", "expected_return_sim", "predicted_class_sim"])

    threshold = float(params.decision_threshold)
    open_trade: Optional[Dict[str, object]] = None
    records: List[Dict[str, object]] = []

    for row in frame.itertuples(index=False):
        ts: pd.Timestamp = row.timestamp
        close_price = float(row.close)
        predicted_class = float(row.predicted_class_sim)
        expected_ret = float(row.expected_return_sim)

        if open_trade is not None:
            side = open_trade["side"]
            exit_signal = False
            if side == "LONG":
                exit_signal = predicted_class in {0.0, -1.0, -2.0}
            elif side == "SHORT":
                exit_signal = predicted_class in {0.0, 1.0, 2.0}

            if exit_signal:
                entry_price = open_trade["entry_price"]
                if side == "LONG":
                    ret = (close_price - entry_price) / entry_price
                else:
                    ret = (entry_price - close_price) / entry_price
                holding_minutes = max((ts - open_trade["entry_time"]).total_seconds() / 60.0, 0.0)
                records.append(
                    {
                        "side": side,
                        "entry_time": open_trade["entry_time"],
                        "exit_time": ts,
                        "entry_price": entry_price,
                        "exit_price": close_price,
                        "return": ret,
                        "holding_mins": holding_minutes,
                        "entry_expected_return": open_trade["entry_expected_return"],
                        "exit_expected_return": expected_ret,
                        "entry_class": open_trade["entry_class"],
                        "exit_class": predicted_class,
                        "exit_reason": "signal_reversal",
                    }
                )
                open_trade = None

        if open_trade is None and expected_ret >= threshold:
            if predicted_class == 2.0:
                open_trade = {
                    "side": "LONG",
                    "entry_time": ts,
                    "entry_price": close_price,
                    "entry_expected_return": expected_ret,
                    "entry_class": predicted_class,
                }
            elif predicted_class == -2.0:
                open_trade = {
                    "side": "SHORT",
                    "entry_time": ts,
                    "entry_price": close_price,
                    "entry_expected_return": expected_ret,
                    "entry_class": predicted_class,
                }

    if open_trade is not None:
        # 在資料結尾強制平倉，避免遺留持倉
        last_row = frame.iloc[-1]
        last_ts: pd.Timestamp = last_row["timestamp"]
        last_price = float(last_row["close"])
        side = open_trade["side"]
        if side == "LONG":
            ret = (last_price - open_trade["entry_price"]) / open_trade["entry_price"]
        else:
            ret = (open_trade["entry_price"] - last_price) / open_trade["entry_price"]
        holding_minutes = max((last_ts - open_trade["entry_time"]).total_seconds() / 60.0, 0.0)
        records.append(
            {
                "side": side,
                "entry_time": open_trade["entry_time"],
                "exit_time": last_ts,
                "entry_price": open_trade["entry_price"],
                "exit_price": last_price,
                "return": ret,
                "holding_mins": holding_minutes,
                "entry_expected_return": open_trade["entry_expected_return"],
                "exit_expected_return": float(last_row["expected_return_sim"]),
                "entry_class": open_trade["entry_class"],
                "exit_class": float(last_row["predicted_class_sim"]),
                "exit_reason": "end_of_data",
            }
        )

    trades = pd.DataFrame.from_records(records, columns=trade_columns)
    if trades.empty:
        return trades

    trades["entry_zscore"] = trades["entry_zscore"].fillna(0.0)
    trades["exit_zscore"] = trades["exit_zscore"].fillna(0.0)
    return trades


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
        side_returns = pd.to_numeric(trades.loc[mask, "return"], errors="coerce").dropna()
        if not side_returns.empty:
            summary[f"{key}_trades"] = float(len(side_returns))
            summary[f"{key}_total_return"] = float(np.prod(1.0 + side_returns) - 1.0)
            summary[f"{key}_win_rate"] = float((side_returns > 0).mean())
            expected_series = pd.to_numeric(
                trades.loc[mask, "entry_expected_return"], errors="coerce"
            ).dropna()
            summary[f"mean_expected_{key}"] = float(expected_series.mean()) if not expected_series.empty else 0.0
        else:
            summary[f"{key}_trades"] = 0.0
            summary[f"{key}_total_return"] = 0.0
            summary[f"{key}_win_rate"] = 0.0
            summary[f"mean_expected_{key}"] = 0.0

    return summary


def _split_train_valid(train_df: pd.DataFrame, valid_days: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame]:
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


def _init_model(params: StarModelParams) -> lgb.LGBMClassifier:
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
        verbosity=-1
    )


def _fit_model(
    model: lgb.LGBMClassifier,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> Tuple[lgb.LGBMClassifier, np.ndarray]:
    eval_set = None
    eval_sample_weight = None
    if not valid_df.empty:
        eval_set = [(valid_df[feature_cols], valid_df[TARGET_COLUMN])]
        eval_sample_weight = [valid_df[SAMPLE_WEIGHT_COLUMN]]
    model.fit(
        train_df[feature_cols],
        train_df[TARGET_COLUMN],
        sample_weight=train_df[SAMPLE_WEIGHT_COLUMN],
        eval_set=eval_set,
        eval_sample_weight=eval_sample_weight,
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
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
) -> Dict[str, float]:
    if df.empty:
        metrics = {
            "accuracy": 0.0,
            "mae_expected": 0.0,
            "threshold": float(params.decision_threshold),
        }
        metrics.update(_summarize_trades(pd.DataFrame()))
        metrics["score"] = 0.0
        return metrics

    expected_returns = _expected_returns(probs, class_means)
    preds_idx = np.argmax(probs, axis=1)
    pred_classes = CLASS_VALUES[preds_idx]
    y_true = df[TARGET_COLUMN].to_numpy()
    accuracy = float(accuracy_score(y_true, pred_classes))
    mae_expected = float(mean_absolute_error(df["future_short_return"], expected_returns))

    trades = _simulate_trades(df, expected_returns, pred_classes, params)
    summary = _summarize_trades(trades)

    metrics = {
        "accuracy": accuracy,
        "mae_expected": mae_expected,
        "threshold": float(params.decision_threshold),
    }
    metrics.update(summary)
    metrics["score"] = metrics.get("total_return", 0.0)
    return metrics


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


def train_star_model(
    dataset: pd.DataFrame,
    indicator_params: StarIndicatorParams,
    model_candidates: Iterable[StarModelParams],
    *,
    model_dir: Path,
    valid_days: int = 30,
) -> StarTrainingResult:
    """針對單一指標參數搜尋最佳模型。"""
    if dataset.empty:
        raise ValueError("提供的資料集為空，無法訓練模型")

    feature_cols = list_feature_columns(dataset)
    if not feature_cols:
        raise ValueError("找不到可用於建模的特徵欄位")

    train_df, test_df = _split_train_valid(dataset, valid_days=valid_days)
    base_df, valid_df = _split_train_valid(train_df, valid_days=min(15, valid_days))
    if base_df.empty:
        base_df = train_df
        valid_df = pd.DataFrame(columns=train_df.columns)

    class_means = _compute_class_means(train_df)

    rankings: List[Dict[str, object]] = []
    best_tuple: Tuple[StarModelParams, Dict[str, float], lgb.LGBMClassifier, np.ndarray] | None = None

    for model_params in model_candidates:
        model = _init_model(model_params)
        fitted_model, val_probs = _fit_model(model, base_df, valid_df, feature_cols)
        val_target_df = valid_df if not valid_df.empty else base_df
        metrics = _evaluate(val_target_df, val_probs, model_params, class_means)
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
    train_probs = final_model.predict_proba(train_df[feature_cols]) if not train_df.empty else np.empty((0, NUM_CLASSES))
    train_metrics = _evaluate(train_df, train_probs, best_model_params, class_means)
    test_probs = (
        final_model.predict_proba(test_df[feature_cols]) if not test_df.empty else np.empty((0, NUM_CLASSES))
    )
    test_metrics = _evaluate(test_df, test_probs, best_model_params, class_means)

    model_dir.mkdir(parents=True, exist_ok=True)
    model_filename = f"star_xgb_{indicator_params.trend_window}_{best_model_params.num_leaves}.txt"
    model_path = model_dir / model_filename
    final_model.booster_.save_model(str(model_path))

    class_thresholds = {
        "q10": float(train_df["q10"].iloc[0]) if "q10" in train_df.columns else 0.0,
        "q25": float(train_df["q25"].iloc[0]) if "q25" in train_df.columns else 0.0,
        "q75": float(train_df["q75"].iloc[0]) if "q75" in train_df.columns else 0.0,
        "q90": float(train_df["q90"].iloc[0]) if "q90" in train_df.columns else 0.0,
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


__all__ = [
    "StarTrainingResult",
    "train_star_model",
    "export_rankings",
    "CLASS_VALUES",
    "_simulate_trades",
    "_summarize_trades",
]
