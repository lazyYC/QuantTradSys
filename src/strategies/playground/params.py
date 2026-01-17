"""star_xgb 策略的參數定義。"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import ClassVar, Dict

from utils.formatting import format_params


@dataclass(frozen=True)
class StarIndicatorParams:
    """控制技術指標與型態偵測的參數。"""

    trend_window: int
    slope_window: int
    atr_window: int
    volatility_window: int
    volume_window: int
    pattern_lookback: int
    upper_shadow_min: float
    body_ratio_max: float
    volume_ratio_max: float
    future_window: int
    future_return_threshold: float
    
    # Momentum / Volatility
    rsi_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_window: int = 20
    bb_std: float = 2.0
    stop_loss_pct: Optional[float] = None # [MODIFIED v1.8.1] Disabled fixed SL. Rely on ATR Trailing.
    
    # New Risk Controls
    adx_threshold: float = 30.0
    max_open_trades: int = 1 # One trade at a time per side per coin? Or keeping max 25 for scale? Let's keep 25.
    max_global_drawdown_pct: float = 0.02
    require_candle_confirmation: bool = True
    
    # v1.8.0 Volatility Breakout Controls
    breakout_window: int = 20      # Lookback for Donchian Channel
    atr_trailing_mult: float = 3.0 # Multiplier for ATR Trailing Stop
    trigger_threshold: float = 0.6 # Prob(Unsafe) to trigger Breakout Mode
    
    # Deprecated Grid Controls
    # grid_step_atr: float = 1.0 
    # max_grid_layers: int = 10    
    # eject_threshold: float = 0.8 
    # suspend_threshold: float = 0.4 
    # trend_filter_ma_window: int = 200 
    # pure_grid: bool = False 


    _ROUND_DECIMALS: ClassVar[Dict[str, int]] = {
        "upper_shadow_min": 2,
        "body_ratio_max": 2,
        "volume_ratio_max": 2,
        "future_return_threshold": 3,
        "bb_std": 1,
        "adx_threshold": 1,
        "max_global_drawdown_pct": 3,
    }

    def as_dict(self, *, rounded: bool = False) -> Dict[str, float | int]:
        data = asdict(self)
        if not rounded:
            return data
        return format_params(data, decimals_map=self._ROUND_DECIMALS)


@dataclass(frozen=True)
class StarModelParams:
    """LightGBM 模型的超參數設定。"""

    num_leaves: int
    max_depth: int
    learning_rate: float
    n_estimators: int
    min_child_samples: int
    subsample: float
    colsample_bytree: float
    feature_fraction_bynode: float
    lambda_l1: float
    lambda_l2: float
    bagging_freq: int
    decision_threshold: float

    _ROUND_DECIMALS: ClassVar[Dict[str, int]] = {
        "learning_rate": 3,
        "subsample": 2,
        "colsample_bytree": 2,
        "feature_fraction_bynode": 2,
        "lambda_l1": 2,
        "lambda_l2": 2,
        "decision_threshold": 4,
    }

    def as_dict(self, *, rounded: bool = False) -> Dict[str, float | int]:
        data = asdict(self)
        if not rounded:
            return data
        return format_params(data, decimals_map=self._ROUND_DECIMALS)


__all__ = ["StarIndicatorParams", "StarModelParams"]
