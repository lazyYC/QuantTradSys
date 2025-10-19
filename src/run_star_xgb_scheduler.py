"""Realtime engine for star_xgb strategy driven by Binance WebSocket closed klines."""
from __future__ import annotations

import argparse
import logging
import threading
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from data_pipeline.binance_ws import BinanceKlineSubscriber
from data_pipeline.ccxt_fetcher import (
    ensure_database,
    fetch_yearly_ohlcv,
    upsert_ohlcv_rows,
)
from notifier.dispatcher import dispatch_signal
from persistence.param_store import load_strategy_params
from persistence.runtime_store import load_runtime_state, save_runtime_state
from strategies.data_utils import prepare_ohlcv_frame, timeframe_to_offset
from strategies.star_xgb.features import StarFeatureCache
from strategies.star_xgb.params import StarIndicatorParams, StarModelParams
from strategies.star_xgb.runtime import StarRuntimeState, generate_realtime_signal, load_star_model

LOGGER = logging.getLogger(__name__)

DEFAULT_LOG_PATH = Path("storage/logs/star_xgb_scheduler.log")
DEFAULT_STATE_DB = Path("storage/strategy_state.db")
DEFAULT_MARKET_DB = Path("storage/market_data.db")

YELLOW = "\033[33m"
RESET = "\033[0m"


def _configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handler = RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    console = logging.StreamHandler()
    handler.setFormatter(formatter)
    console.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    root.addHandler(console)


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


class StarRealtimeEngine:
    """Manage WebSocket ingestion and strategy evaluation."""

    def __init__(
        self,
        *,
        symbol: str,
        timeframe: str,
        strategy: str,
        lookback_days: int,
        params_store_path: Path,
        state_store_path: Path,
        market_db_path: Path,
        exchange: str,
        exchange_config: Optional[Dict] = None,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.strategy = strategy
        self.lookback_days = lookback_days
        self.params_store_path = params_store_path
        self.state_store_path = state_store_path
        self.market_db_path = market_db_path
        self.exchange = exchange
        self.exchange_config = exchange_config

        record = load_strategy_params(self.params_store_path, strategy, symbol, timeframe)
        if record is None or not isinstance(record.params, dict):
            raise RuntimeError(f"策略 {strategy} 缺少已儲存參數")

        payload = record.params
        indicator_payload = payload.get("indicator")
        model_payload = payload.get("model")
        model_path = payload.get("model_path")
        feature_columns = payload.get("feature_columns")
        class_means = payload.get("class_means")
        if not all([indicator_payload, model_payload, model_path, feature_columns, class_means]):
            raise RuntimeError(f"策略 {strategy} 儲存參數不完整")

        self.indicator = StarIndicatorParams(**indicator_payload)
        self.model_params = StarModelParams(**model_payload)
        self.feature_columns = feature_columns
        self.class_means = class_means
        self.model = load_star_model(model_path)

        runtime_record = load_runtime_state(self.state_store_path, strategy, symbol, timeframe)
        self.runtime_state = StarRuntimeState.from_dict(runtime_record.state if runtime_record else None)

        LOGGER.info("初始化 OHLCV 資料 (REST)")
        raw_df = fetch_yearly_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            lookback_days=lookback_days,
            exchange_id=exchange,
            exchange_config=exchange_config,
            prune_history=False,
            db_path=market_db_path,
        )
        prepared = prepare_ohlcv_frame(raw_df, timeframe)
        if prepared.empty:
            raise RuntimeError(f"{symbol} {timeframe} 無歷史 OHLCV 資料")

        self.data_lock = threading.Lock()
        self.db_conn = ensure_database(market_db_path)

        self.price_df = prepared.reset_index(drop=True)

        bars_per_day = int(pd.Timedelta(days=1) / timeframe_to_offset(timeframe))
        self.max_rows = max(int(lookback_days * bars_per_day * 1.1), len(self.price_df))

        self.subscriber: Optional[BinanceKlineSubscriber] = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        LOGGER.info(
            "啟動 star_xgb websocket 引擎 | strategy=%s symbol=%s timeframe=%s lookback=%sd",
            self.strategy,
            self.symbol,
            self.timeframe,
            self.lookback_days,
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
            LOGGER.info("收到中斷訊號，準備關閉...")
        finally:
            if self.subscriber:
                self.subscriber.stop()
            self.db_conn.close()
            LOGGER.info("star_xgb websocket 引擎已關閉")

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

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
            upsert_ohlcv_rows(
                self.db_conn,
                self.symbol,
                self.timeframe,
                [(ts_open, iso_ts, open_, high, low, close, volume)],
            )
        LOGGER.info("收到封棒: %s close=%s volume=%s", ts.isoformat(), close, volume)
        self._evaluate("websocket")

    def _evaluate(self, source: str) -> None:
        with self.data_lock:
            df = self.price_df.copy()
        cache = StarFeatureCache(
            df,
            trend_windows=[self.indicator.trend_window],
            atr_windows=[self.indicator.atr_window],
            volatility_windows=[self.indicator.volatility_window],
            volume_windows=[self.indicator.volume_window],
            pattern_windows=[self.indicator.pattern_lookback],
        )
        action, context, new_state = generate_realtime_signal(
            df=df,
            indicator_params=self.indicator,
            model_params=self.model_params,
            model=self.model,
            feature_columns=self.feature_columns,
            class_means=self.class_means,
            cache=cache,
            state=self.runtime_state,
        )
        context.update(
            {
                "strategy": self.strategy,
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "source": source,
                "received_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        colored_action = f"{YELLOW}{action}{RESET}"
        LOGGER.info("star_xgb action=%s details=%s", colored_action, context)
        if action != "HOLD":
            dispatch_signal(action, context)

        if new_state != self.runtime_state:
            save_runtime_state(
                self.state_store_path,
                strategy=self.strategy,
                symbol=self.symbol,
                timeframe=self.timeframe,
                state=new_state.to_dict(),
            )
            self.runtime_state = new_state


def main() -> None:
    parser = argparse.ArgumentParser(description="star_xgb realtime engine (websocket driven)")
    parser.add_argument("--symbol", type=str, default="BTC/USDT")
    parser.add_argument("--timeframe", type=str, default="5m")
    parser.add_argument("--strategy", type=str, default="star_xgb_default", help="策略名稱 (study name)")
    parser.add_argument("--lookback-days", type=int, default=30, help="初始化時載入的歷史天數")
    parser.add_argument("--params-db", type=Path, default=DEFAULT_STATE_DB)
    parser.add_argument("--state-db", type=Path, default=DEFAULT_STATE_DB)
    parser.add_argument("--ohlcv-db", type=Path, default=DEFAULT_MARKET_DB)
    parser.add_argument("--exchange", type=str, default="binance")
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    args = parser.parse_args()

    _configure_logging(args.log_path)

    engine = StarRealtimeEngine(
        symbol=args.symbol,
        timeframe=args.timeframe,
        strategy=args.strategy,
        lookback_days=args.lookback_days,
        params_store_path=args.params_db,
        state_store_path=args.state_db,
        market_db_path=args.ohlcv_db,
        exchange=args.exchange,
    )
    engine.start()


if __name__ == "__main__":
    main()
