
from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Any

import pandas as pd

from config.paths import DEFAULT_LOG_DIR
from config.env import load_env

from data_pipeline.binance_ws import BinanceKlineSubscriber
from data_pipeline.ccxt_fetcher import fetch_yearly_ohlcv
from persistence.market_store import MarketDataStore
from notifier.dispatcher import dispatch_signal
from persistence.param_store import load_strategy_params, StrategyRecord
from persistence.runtime_store import load_runtime_state, save_runtime_state
from utils.data_utils import prepare_ohlcv_frame, timeframe_to_offset
from utils.symbols import canonicalize_symbol
from utils.pid_lock import PIDLock, AlreadyRunningError

LOGGER = logging.getLogger(__name__)

def _safe_concat(base: pd.DataFrame, row: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if base.empty:
        result = row.reset_index(drop=True)
    else:
        if base["timestamp"].iloc[-1] == row["timestamp"].iloc[0]:
            base = base.copy()
            base.iloc[-1] = row.iloc[0]
            result = base
        else:
            result = pd.concat([base, row], ignore_index=True)
    if max_rows and len(result) > max_rows:
        result = result.iloc[-max_rows:].reset_index(drop=True)
    return result


class RealtimeEngine:
    """Manage WebSocket ingestion and strategy evaluation."""

    def __init__(
        self,
        *,
        symbol: str,
        timeframe: str,
        strategy: str,
        study: str,
        lookback_days: int,

        exchange: str,
        exchange_config: Optional[Dict] = None,
        strategy_record: Optional[StrategyRecord] = None,
    ) -> None:
        self.symbol = canonicalize_symbol(symbol)
        self.timeframe = timeframe
        self.strategy = strategy
        self.study = study
        self.lookback_days = lookback_days

        self.exchange = exchange
        self.exchange_config = exchange_config

        # Use injected record or load it
        if strategy_record:
            record = strategy_record
        else:
            record = load_strategy_params(
                strategy=strategy, 
                study=study, 
                symbol=self.symbol, 
                timeframe=timeframe
            )

        if record is None or not isinstance(record.params, dict):
            raise RuntimeError(f"Parameters not found for strategy={strategy} study={study}")
            
        payload = record.params
        # Inject model_path from record if not in payload
        if "model_path" not in payload and record.model_path:
            payload["model_path"] = record.model_path
        
        # Validate Strategy Params
        # Dynamic Runtime Loading
        from strategies.loader import load_strategy_runtime
        runtime_mod = load_strategy_runtime(strategy)
        
        # Bind generic hooks
        self.RuntimeStateCls = getattr(runtime_mod, "StarRuntimeState", getattr(runtime_mod, "RuntimeState", None))
        if not self.RuntimeStateCls:
             raise RuntimeError(f"Runtime {strategy} must define ReferenceState or RuntimeState")
        
        self.generate_signal_fn = runtime_mod.generate_realtime_signal
        
        # load_model alias?
        if hasattr(runtime_mod, "load_star_model"):
            self.load_model_fn = runtime_mod.load_star_model
        elif hasattr(runtime_mod, "load_model"):
             self.load_model_fn = runtime_mod.load_model
        else:
             raise RuntimeError(f"Runtime {strategy} missing load_model function")
             
        # Initialize params using the loaded module
        self._init_strategy_params(payload, runtime_mod)

        self.timeframe_offset = timeframe_to_offset(self.timeframe)
        
        # Load Runtime State
        runtime_record = load_runtime_state(
            strategy=strategy, 
            study=study, 
            symbol=self.symbol, 
            timeframe=timeframe
        )
        
        if self.RuntimeStateCls:
            self.runtime_state = self.RuntimeStateCls.from_dict(
                runtime_record.state if runtime_record else None
            )
            # Restore min_exit_timestamp if needed
            if (
                self.runtime_state.position_side
                and self.runtime_state.entry_timestamp is not None
                and self.min_hold_duration is not None
                and self.runtime_state.min_exit_timestamp is None
            ):
                self.runtime_state.min_exit_timestamp = (
                    self.runtime_state.entry_timestamp + self.min_hold_duration
                )

        LOGGER.info("Initializing OHLCV data (REST)...")
        raw_df = fetch_yearly_ohlcv(
            symbol=self.symbol,
            timeframe=timeframe,
            lookback_days=lookback_days,
            exchange_id=exchange,
            exchange_config=exchange_config,
            prune_history=False,
            # db_path=market_db_path, # Removed
        )
        prepared = prepare_ohlcv_frame(raw_df, timeframe)
        if prepared.empty:
            raise RuntimeError(f"No historical OHLCV data for {self.symbol} {timeframe}")

        self.data_lock = threading.Lock()
        self.market_store = MarketDataStore()
        # self.db_conn = ensure_database(market_db_path) # Removed

        self.price_df = prepared.reset_index(drop=True)

        bars_per_day = int(pd.Timedelta(days=1) / timeframe_to_offset(timeframe))
        self.max_rows = max(int(lookback_days * bars_per_day * 1.1), len(self.price_df))

        self.subscriber: Optional[BinanceKlineSubscriber] = None

    def _init_strategy_params(self, payload: Dict, runtime_mod: Any) -> None:
        # Extract params for Indicator and Model from flat payload
        # We filter the payload to match the dataclass fields
        from dataclasses import fields
        
        # Use aliases if available, else fallback
        IndicatorParamsCls = getattr(runtime_mod, "IndicatorParams", getattr(runtime_mod, "StarIndicatorParams"))
        ModelParamsCls = getattr(runtime_mod, "ModelParams", getattr(runtime_mod, "StarModelParams"))
        
        indicator_keys = {f.name for f in fields(IndicatorParamsCls)}
        model_keys = {f.name for f in fields(ModelParamsCls)}
        
        indicator_kwargs = {k: v for k, v in payload.items() if k in indicator_keys}
        model_kwargs = {k: v for k, v in payload.items() if k in model_keys}
        
        model_path = payload.get("model_path")
        feature_columns = payload.get("feature_columns")
        class_means = payload.get("class_means")
        feature_stats = payload.get("feature_stats")
        
        # Check for missing required keys
        missing_indicator = indicator_keys - indicator_kwargs.keys()
        missing_model = model_keys - model_kwargs.keys()
        
        if missing_indicator:
             raise RuntimeError(f"Missing indicator params: {missing_indicator}")
        if missing_model:
             raise RuntimeError(f"Missing model params: {missing_model}")
             
        if not all([model_path, feature_columns, class_means, feature_stats]):
             raise RuntimeError(f"Missing artifacts (model_path, feature_columns, etc) for strategy={self.strategy} study={self.study}")
             
        self.indicator = IndicatorParamsCls(**indicator_kwargs)
        self.model_params = ModelParamsCls(**model_kwargs)
        self.feature_columns = feature_columns
        self.class_means = class_means
        self.feature_stats = feature_stats
        self.model = self.load_model_fn(model_path)
        
        self.min_hold_bars = max(int(self.indicator.future_window), 0)
        self.min_hold_duration = (
            timeframe_to_offset(self.timeframe) * self.min_hold_bars
            if self.min_hold_bars > 0
            else None
        )
        self.stop_loss_pct = self._resolve_stop_loss_pct(payload.get("stop_loss_pct"))
        
        if self.min_hold_duration:
            LOGGER.info("Runtime min hold: bars=%s duration=%s", self.min_hold_bars, self.min_hold_duration)
        else:
            LOGGER.info("Runtime min hold disabled")
            

        if self.stop_loss_pct:
            LOGGER.info("Runtime stop loss: %.4f", self.stop_loss_pct)

        # Resolve Trading Config for Display
        self.order_ratio = self._resolve_order_ratio(self.exchange)
        self.leverage = self._resolve_leverage(self.exchange)

    def start(self) -> None:
        LOGGER.info(
            "Starting Realtime Engine | strategy=%s study=%s symbol=%s timeframe=%s lookback=%sd",
            self.strategy,
            self.study,
            self.symbol,
            self.timeframe,
            self.lookback_days,
        )
        LOGGER.info(
            "Trading Config | Exchange=%s Leverage=%sx Order Ratio=%.2f",
            self.exchange.upper(),
            self.leverage,
            self.order_ratio,
        )
        self._evaluate("initial")

        self.subscriber = BinanceKlineSubscriber(
            symbol=self.symbol,
            timeframe=self.timeframe,
            on_closed_kline=self._handle_closed_kline,
        )
        self.subscriber.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            LOGGER.info("Interrupt received, shutting down...")
        finally:
            if self.subscriber:
                self.subscriber.stop()
            # self.db_conn.close() # No explicit close needed for managed session
            LOGGER.info("Realtime Engine stopped")

    def _handle_closed_kline(self, kline: dict) -> None:
        ts_open = int(kline["t"])
        open_ = float(kline["o"])
        high = float(kline["h"])
        low = float(kline["l"])
        close = float(kline["c"])
        volume = float(kline["v"])
        ts = datetime.fromtimestamp(ts_open / 1000, tz=timezone.utc)
        iso_ts = ts.isoformat().replace("+00:00", "Z")

        row_df = pd.DataFrame(
            {
                "timestamp": [ts],
                "open": [open_],
                "high": [high],
                "low": [low],
                "close": [close],
                "volume": [volume],
            }
        )

        with self.data_lock:
            self.price_df = _safe_concat(self.price_df, row_df, self.max_rows)
            self.market_store.upsert_candles(
                data=[(ts_open, iso_ts, open_, high, low, close, volume)],
                symbol=self.symbol,
                timeframe=self.timeframe
            )
        LOGGER.info("Closed Kline: %s close=%s volume=%s", ts.isoformat(), close, volume)
        self._evaluate("websocket")

    def _evaluate(self, source: str) -> None:
        self._evaluate_strategy(source)

    def _evaluate_strategy(self, source: str) -> None:
        with self.data_lock:
            df = self.price_df.copy()
            
        # OCP: We do NOT build explicit cache here, relying on runtime to build it if needed.
        # This removes the dependency on StarFeatureCache.
        
        action, context, new_state = self.generate_signal_fn(
            df=df,
            indicator_params=self.indicator,
            model_params=self.model_params,
            model=self.model,
            feature_columns=self.feature_columns,
            class_means=self.class_means,
            feature_stats=self.feature_stats,
            cache=None, # runtime will build it
            state=self.runtime_state,
            min_hold_duration=self.min_hold_duration,
            stop_loss_pct=self.stop_loss_pct,
        )
        
        context.update(
            {
                "strategy": self.strategy,
                "study": self.study,
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "received_at": datetime.now(timezone.utc).isoformat(),
                "leverage": self.leverage,
            }
        )
        
        colored_action = self.assign_color(action)
        context_filtered = self.filter_context(context)
        context_filtered["action"] = colored_action
        
        context_str = "\n".join([f"{k}: {v}" for k, v in context_filtered.items()])
        context_str += "\n--------------------------------"
        LOGGER.info(f"{context_str}")
        
        trade_executed = True
        if action != "HOLD":
            trade_executed = dispatch_signal(action, context)
            if not trade_executed:
                warning_msg = f"Dispatch reported no execution for action={action}; runtime state unchanged"
                colored_warning_msg = self.assign_color(warning_msg, color="RED")
                LOGGER.warning(colored_warning_msg)

        if trade_executed and new_state != self.runtime_state:
            save_runtime_state(
                strategy=self.strategy,
                study=self.study,
                symbol=self.symbol,
                timeframe=self.timeframe,
                state=new_state.to_dict(),
            )
            self.runtime_state = new_state

    # ------------------------------------------------------------------ #
    # Internal utilities
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_stop_loss_pct(stored_value: Optional[float]) -> Optional[float]:
        env_keys = ["STAR_STOP_LOSS_PCT", "STOP_LOSS_PCT"]
        for key in env_keys:
            raw = os.getenv(key)
            if raw is None:
                continue
            try:
                value = float(raw)
            except ValueError:
                LOGGER.warning("Invalid %s value %s; ignoring", key, raw)
                continue
            if value <= 0:
                return None
            return value

        if stored_value is not None:
            try:
                value = float(stored_value)
            except (TypeError, ValueError):
                LOGGER.debug("Stored stop_loss_pct %s invalid; ignoring", stored_value)
            else:
                return value if value > 0 else None

        return 0.01

    @staticmethod
    def _resolve_order_ratio(exchange: str) -> float:
        # Default to 0.97 if not set (legacy behavior from dispatcher)
        default_ratio = 0.97
        
        # Map exchange to broker prefix
        broker_prefix = "BINANCE" if "binance" in exchange.lower() else "ALPACA"
        
        env_keys = [
            f"{broker_prefix}_ORDER_RATIO",
            "ORDER_RATIO",
        ]
        if broker_prefix != "ALPACA":
            env_keys.append("ALPACA_ORDER_RATIO") # Fallback

        for key in env_keys:
            raw = os.getenv(key)
            if raw is None:
                continue
            try:
                value = max(min(float(raw), 1.0), 0.0)
                return value
            except ValueError:
                pass
        return default_ratio

    @staticmethod
    def _resolve_leverage(exchange: str) -> int:
        # Default to 1 (No leverage)
        default_leverage = 1
        
        # Map exchange to broker prefix
        broker_prefix = "BINANCE" if "binance" in exchange.lower() else "ALPACA"
        
        env_keys = [
            f"{broker_prefix}_LEVERAGE",
            "LEVERAGE",
        ]

        for key in env_keys:
            raw = os.getenv(key)
            if raw is None:
                continue
            try:
                value = int(raw)
                return value if value > 0 else default_leverage
            except ValueError:
                pass
        return default_leverage

    @staticmethod
    def filter_context(context: Dict[str, object]) -> Dict[str, object]:
        exclude_keys = ["threshold", "hold_elapsed", "timeframe", "min_exit_timestamp", "source", ]
        return {k: v for k, v in context.items() if k not in exclude_keys}

    @staticmethod
    def assign_color(action: str, **kwargs) -> str:
        color_map = {
            "GREEN": "\033[32m",
            "RED": "\033[31m",
            "YELLOW": "\033[33m",
            "RESET": "\033[0m",
        }

        if kwargs.get("color"):
            return f"{color_map[kwargs['color']]}{action}{color_map['RESET']}"

        if action == "ENTER_LONG":
            return f"{color_map['GREEN']}ENTER_LONG{color_map['RESET']}"
        elif action == "ENTER_SHORT":
            return f"{color_map['RED']}ENTER_SHORT{color_map['RESET']}"
        elif action == "EXIT_LONG":
            return f"{color_map['GREEN']}EXIT_LONG{color_map['RESET']}"
        elif action == "EXIT_SHORT":
            return f"{color_map['RED']}EXIT_SHORT{color_map['RESET']}"
        elif action == "NET_ZERO":
            return f"{color_map['YELLOW']}NET_ZERO{color_map['RESET']}"
        return f"{color_map['YELLOW']}HOLD{color_map['RESET']}"
