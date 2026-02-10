"""
Parameter dataclasses for Meta-Labeling strategy.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, Any


@dataclass
class PrimarySignalParams:
    """Parameters for primary signal generator (Layer 1)."""
    
    # RSI Parameters
    rsi_window: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    
    # MA Cross Parameters
    fast_ma_window: int = 10
    slow_ma_window: int = 30
    
    # Bollinger Band Parameters
    bb_window: int = 20
    bb_std: float = 2.0
    
    # Signal combination mode: "any" (OR) or "all" (AND)
    signal_mode: str = "any"
    
    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TripleBarrierParams:
    """Parameters for Triple Barrier labeling."""
    
    # Barrier Type
    use_atr: bool = True       # Use ATR for dynamic width
    atr_window: int = 20       # ATR lookback
    
    # ATR Multipliers (if use_atr=True)
    tp_atr_mult: float = 2.0
    sl_atr_mult: float = 2.0
    
    # Fixed Percentages (if use_atr=False)
    # Note: TP < SL ratio favors shorter trades hitting profit first
    take_profit_pct: float = 0.003  # 0.3%
    stop_loss_pct: float = 0.005    # 0.5%
    
    # Minimum profit threshold (to cover transaction costs)
    # 0.001 entry + 0.001 exit = 0.002 cost.
    # We require profit > 0.002 to be labeled as 1.
    min_profit_pct: float = 0.002
    
    # Vertical barrier (max holding time in bars)
    max_holding_bars: int = 120  # Reduced to 2 hours
    
    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MetaModelParams:
    """Parameters for meta-model (Layer 2)."""
    
    # LightGBM parameters
    num_leaves: int = 31
    max_depth: int = 6
    learning_rate: float = 0.05
    n_estimators: int = 200
    min_child_samples: int = 20
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    
    # Meta threshold for trade execution
    meta_threshold: float = 0.5
    
    # Feature lookback
    volatility_window: int = 20
    volume_window: int = 20
    
    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)
