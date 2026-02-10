"""
Meta-Model Training and Inference for Meta-Labeling Strategy.
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)

from strategies.metalabel.params import MetaModelParams
from strategies.metalabel.features import build_meta_features, get_feature_columns

LOGGER = logging.getLogger(__name__)


@dataclass
class MetaModelResult:
    """Result from meta-model training."""
    model_path: Path
    validation_metrics: Dict[str, float]
    feature_importance: pd.DataFrame
    feature_columns: list


def train_meta_model(
    labeled_df: pd.DataFrame,
    features_df: pd.DataFrame,
    params: MetaModelParams,
    model_dir: str,
    seed: int = 42,
) -> MetaModelResult:
    """
    Train the meta-model (Layer 2).
    
    Args:
        labeled_df: DataFrame with signal bars and meta_labels
        features_df: DataFrame with pre-computed meta features
        params: Meta model parameters
        model_dir: Directory to save model
        seed: Random seed
        
    Returns:
        MetaModelResult with model path and metrics
    """
    LOGGER.info(f"Training meta-model on {len(labeled_df)} labeled samples...")
    
    # Features are already built and passed in features_df
    all_features = features_df
    
    # Extract features for labeled samples
    feature_cols = get_feature_columns()
    X = all_features.loc[labeled_df.index, feature_cols]
    y = labeled_df["meta_label"].values
    
    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y if len(np.unique(y)) > 1 else None
    )
    
    LOGGER.info(f"Training set: {len(X_train)}, Validation set: {len(X_val)}")
    LOGGER.info(f"Label distribution - Train: {np.bincount(y_train.astype(int))}, Val: {np.bincount(y_val.astype(int))}")
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # LightGBM parameters
    lgb_params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "boosting_type": "gbdt",
        "num_leaves": params.num_leaves,
        "max_depth": params.max_depth,
        "learning_rate": params.learning_rate,
        "min_child_samples": params.min_child_samples,
        "reg_alpha": params.reg_alpha,
        "reg_lambda": params.reg_lambda,
        "verbose": -1,
        "seed": seed,
    }
    
    # Train model
    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=params.n_estimators,
        valid_sets=[train_data, val_data],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30),
            lgb.log_evaluation(period=-1),  # Silence per-iteration logs
        ],
    )
    
    # Predict on validation
    y_pred_proba = model.predict(X_val)
    y_pred = (y_pred_proba >= params.meta_threshold).astype(int)
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, zero_division=0),
        "recall": recall_score(y_val, y_pred, zero_division=0),
        "f1": f1_score(y_val, y_pred, zero_division=0),
    }
    
    if len(np.unique(y_val)) > 1:
        try:
            metrics["auc"] = roc_auc_score(y_val, y_pred_proba)
        except Exception:
            metrics["auc"] = 0.5
    else:
        metrics["auc"] = 0.5
    
    # Minimal logging
    LOGGER.info(
        f"Validation Metrics: F1={metrics['f1']:.4f}, AUC={metrics.get('auc', 0):.4f}, "
        f"Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}"
    )
    
    # Feature importance
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    
    # Save model
    model_path = Path(model_dir) / "meta_model.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump({
        "model": model,
        "feature_columns": feature_cols,
        "params": params,
    }, model_path)
    
    return MetaModelResult(
        model_path=model_path,
        validation_metrics=metrics,
        feature_importance=importance,
        feature_columns=feature_cols,
    )


def load_meta_model(model_path: str) -> Tuple[lgb.Booster, list, MetaModelParams]:
    """Load a trained meta-model."""
    data = joblib.load(model_path)
    return data["model"], data["feature_columns"], data["params"]


def predict_meta_probability(
    model: lgb.Booster,
    features: pd.DataFrame,
    feature_columns: list,
) -> np.ndarray:
    """Predict probability of profitable trade."""
    X = features[feature_columns]
    return model.predict(X)
