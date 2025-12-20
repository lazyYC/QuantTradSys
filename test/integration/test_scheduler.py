
import os
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone
import pandas as pd

from engine.realtime import RealtimeEngine
from persistence.param_store import StrategyRecord
from notifier.dispatcher import dispatch_signal

@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setenv("BINANCE_LEVERAGE", "5")
    monkeypatch.setenv("BINANCE_ORDER_RATIO", "0.95")
    monkeypatch.setenv("STOP_LOSS_PCT", "0.02")

@pytest.fixture
def dummy_strategy_record():
    return StrategyRecord(
        strategy="playground",
        study="ver_test",
        symbol="BTC/USDT:USDT",
        timeframe="1h",
        params={
            "feature_columns": ["close", "volume"],
            "class_means": [[0.1, 0.2]],
            "feature_stats": {"mean": 0, "std": 1},
            "model_path": "dummy_model.pkl",
            "future_window": 12, # 1h * 12
        },
        metrics={},
        model_path="dummy_model.pkl",
        updated_at=datetime.now(timezone.utc).isoformat()
    )

@pytest.fixture
def mock_runtime_module():
    # Mock the strategy runtime module
    mock_mod = MagicMock()
    # Mock RuntimeState
    mock_state_cls = MagicMock()
    mock_state_cls.from_dict.return_value.position_side = None
    mock_state_cls.from_dict.return_value.stacks = []
    mock_mod.StarRuntimeState = mock_state_cls
    
    # Mock generate_realtime_signal
    # Returns: action, context, new_state
    mock_state = MagicMock()
    mock_state.to_dict.return_value = {"mock_state": "valid"}
    
    mock_mod.generate_realtime_signal.return_value = (
        "ENTER_LONG", 
        {"price": 50000.0, "scale": 1.0}, 
        mock_state
    )
    
    # Mock load_model
    mock_mod.load_star_model.return_value = MagicMock()
    
    # Define dummy dataclasses
    from dataclasses import dataclass, field
    from typing import List, Optional
    
    @dataclass
    class DummyIndicatorParams:
        future_window: int = 12

    @dataclass
    class DummyModelParams:
        model_path: Optional[str] = None
        feature_columns: Optional[List[str]] = None
        class_means: Optional[List[List[float]]] = None
        feature_stats: Optional[dict] = None

    mock_mod.IndicatorParams = DummyIndicatorParams
    mock_mod.ModelParams = DummyModelParams
    
    return mock_mod

@patch("strategies.loader.load_strategy_runtime")
@patch("engine.realtime.fetch_yearly_ohlcv")
@patch("engine.realtime.prepare_ohlcv_frame")
@patch("engine.realtime.ensure_database")
@patch("engine.realtime.load_runtime_state")
@patch("engine.realtime.dispatch_signal")  # We want to verify this call
def test_scheduler_flow_and_leverage(
    mock_dispatch,
    mock_load_state,
    mock_ensure_db,
    mock_prepare,
    mock_fetch,
    mock_load_runtime,
    mock_env,
    dummy_strategy_record,
    mock_runtime_module
):
    # Setup Mocks
    mock_load_runtime.return_value = mock_runtime_module
    
    # Mock OHLCV data
    dummy_df = pd.DataFrame({
        "timestamp": [pd.Timestamp.now(timezone.utc)],
        "open": [100.0], "high": [110.0], "low": [90.0], "close": [105.0], "volume": [1000.0]
    })
    mock_fetch.return_value = dummy_df
    mock_prepare.return_value = dummy_df # Return same df
    
    # Initialize Engine
    engine = RealtimeEngine(
        symbol="BTC/USDT:USDT",
        timeframe="1h",
        strategy="playground",
        study="ver_test",
        lookback_days=1,
        params_store_path=Path("dummy_params.db"),
        state_store_path=Path("dummy_state.db"),
        market_db_path=Path("dummy_market.db"),
        exchange="binanceusdm",
        strategy_record=dummy_strategy_record
    )
    
    # Test Initialization Logic
    assert engine.leverage == 5
    assert engine.order_ratio == 0.95
    assert engine.stop_loss_pct == 0.02
    
    # Test Step Execution (Simulate "Closed Kline")
    # We call _evaluate directly or mock the subscriber. 
    # _evaluate calls generate_signal_fn -> dispatch_signal
    
    # We need to set engine.runtime_state to not None (done by init via mock_load_state)
    # But mock_load_state returns None by default? let's make it return a record or None
    mock_load_state.return_value = None # Should trigger fresh init
    
    # To run step, we need a df row.
    row = dummy_df.iloc[0]
    
    # Mock the internal generate_signal_fn to return ENTER_LONG
    # (Actually it calls the mocked runtime module function)
    
    engine._evaluate("test") # This calls generate_signal_fn
    
    # VERIFICATION
    # Check if dispatch_signal was called
    mock_dispatch.assert_called_once()
    
    # Check arguments: action, context
    args, kwargs = mock_dispatch.call_args
    action = args[0]
    context = args[1]
    
    assert action == "ENTER_LONG"
    assert context["leverage"] == 5
    assert context["symbol"] == "BTC/USDT:USDT"
    print("\nVerified: Context contains leverage=5")

if __name__ == "__main__":
    # Allow running directly for quick check
    sys.exit(pytest.main(["-v", __file__]))
