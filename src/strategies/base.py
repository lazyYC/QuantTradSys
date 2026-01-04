from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import optuna
import pandas as pd


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    All strategies (XGBoost, Deep Learning, etc.) must implement this interface.
    """

    @property
    def can_backtest_dev_data(self) -> bool:
        """
        Whether the strategy supports time-continuous backtesting on development (Train/Valid) data.
        Defaults to True. Set to False for strategies that use random shuffling (e.g. Playground).
        """
        return True

    @abstractmethod
    def build_features(
        self, raw_data: pd.DataFrame, params: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Construct features from raw OHLCV data.

        Args:
            raw_data: DataFrame containing 'timestamp', 'open', 'high', 'low', 'close', 'volume'.
            params: Dictionary of parameters for feature engineering (e.g., window sizes).

        Returns:
            DataFrame with calculated features.
        """
        pass

    @abstractmethod
    def train(
        self,
        train_data: pd.DataFrame,
        valid_data: Optional[pd.DataFrame],
        params: Dict[str, Any],
        model_dir: str,
        seed: int,
    ) -> Any:
        """
        Train the model.

        Args:
            train_data: Training dataset (features + labels).
            valid_data: Validation dataset (optional).
            params: Dictionary of model hyperparameters.
            model_dir: Directory to save model artifacts.
            seed: Random seed for reproducibility.

        Returns:
            The trained model object (or a result object containing the model).
        """
        pass

    @abstractmethod
    def get_optuna_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Define the hyperparameter search space for Optuna.

        Args:
            trial: The current Optuna trial.

        Returns:
            A dictionary of suggested parameters for this trial.
        """
        pass

    @abstractmethod
    def warm_up(self, raw_data: pd.DataFrame) -> None:
        """
        Pre-calculate expensive features or initialize caches.
        This is called once before the optimization loop.
        
        Args:
            raw_data: The raw OHLCV data.
        """
        pass

    @abstractmethod
    def prepare_data(
        self, raw_data: pd.DataFrame, params: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Prepare data for training (feature engineering, labeling, splitting).
        
        Args:
            raw_data: The raw OHLCV data.
            params: Configuration parameters (including indicator params).
            
        Returns:
            Tuple of (prepared_dataset, metadata).
            metadata can contain things like feature_columns, class_means, etc.
        """
        pass

    @abstractmethod
    def backtest(
        self, 
        raw_data: pd.DataFrame, 
        params: Dict[str, Any], 
        model_path: Optional[str] = None
    ) -> Any:
        """
        Run backtest using the provided parameters and optional model.
        
        Args:
            raw_data: The raw OHLCV data.
            params: Strategy parameters.
            model_path: Path to the trained model (if applicable).
            
        Returns:
            A result object containing trades, metrics, and equity curve.
        """
        pass

    def split_data(
        self, 
        data: pd.DataFrame, 
        test_days: int = 30, 
        valid_days: int = 30
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, valid, and test sets.
        Default implementation is time-based (Hold-out).
        
        Args:
            data: Prepared DataFrame (usually containing timestamp).
            test_days: Number of days for the test set (end of data).
            valid_days: Number of days for the validation set (before test set).
            
        Returns:
            (train_df, valid_df, test_df)
        """
        if data.empty:
            return data.copy(), data.copy(), data.copy()
            
        max_timestamp = data["timestamp"].max()
        test_cutoff = max_timestamp - pd.Timedelta(days=test_days)
        valid_cutoff = test_cutoff - pd.Timedelta(days=valid_days)
        
        test_df = data[data["timestamp"] > test_cutoff].copy()
        valid_df = data[(data["timestamp"] > valid_cutoff) & (data["timestamp"] <= test_cutoff)].copy()
        train_df = data[data["timestamp"] <= valid_cutoff].copy()
        
        return train_df.reset_index(drop=True), valid_df.reset_index(drop=True), test_df.reset_index(drop=True)

