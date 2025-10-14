"""啟動 star_xgb 策略的即時訊號排程。"""
from __future__ import annotations

import argparse
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from apscheduler.schedulers.blocking import BlockingScheduler

from data_pipeline.ccxt_fetcher import fetch_yearly_ohlcv
from notifier.dispatcher import dispatch_signal
from persistence.param_store import StrategyRecord, load_strategy_params
from persistence.runtime_store import load_runtime_state, save_runtime_state
from strategies.data_utils import prepare_ohlcv_frame
from strategies.star_xgb.features import StarFeatureCache
from strategies.star_xgb.params import StarIndicatorParams, StarModelParams
from strategies.star_xgb.runtime import StarRuntimeState, generate_realtime_signal, load_star_model

LOGGER = logging.getLogger(__name__)

DEFAULT_LOG_PATH = Path("storage/logs/star_xgb_scheduler.log")
DEFAULT_STATE_DB = Path("storage/strategy_state.db")

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


def run_star_cycle(
    symbol: str,
    timeframe: str,
    strategy: str,
    # *,
    lookback_days: int,
    params_store_path: Path,
    state_store_path: Path,
    exchange: str,
    exchange_config: Optional[Dict] = None,
) -> Dict[str, object]:
    record = load_strategy_params(params_store_path, strategy, symbol, timeframe)
    if record is None or not isinstance(record.params, dict):
        LOGGER.warning("策略 %s 的參數不存在或格式不正確。", strategy)
        return {"action": "HOLD", "reason": "missing_params"}

    payload = record.params
    indicator_payload = payload.get("indicator")
    model_payload = payload.get("model")
    model_path = payload.get("model_path")
    feature_columns = payload.get("feature_columns")
    class_means = payload.get("class_means")

    if not all([indicator_payload, model_payload, model_path, feature_columns, class_means]):
        LOGGER.warning("策略 %s 的參數不完整。", strategy)
        return {"action": "HOLD", "reason": "incomplete_params"}

    indicator = StarIndicatorParams(**indicator_payload)
    model_params = StarModelParams(**model_payload)
    booster = load_star_model(model_path)

    raw_df = fetch_yearly_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        lookback_days=lookback_days,
        exchange_id=exchange,
        exchange_config=exchange_config,
        prune_history=False,
    )
    cleaned = prepare_ohlcv_frame(raw_df, timeframe)
    if cleaned.empty:
        LOGGER.warning("無法取得 %s %s 的 OHLCV 資料。", symbol, timeframe)
        return {"action": "HOLD", "reason": "empty_data"}

    runtime_record = load_runtime_state(state_store_path, strategy, symbol, timeframe)
    runtime_state = StarRuntimeState.from_dict(runtime_record.state if runtime_record else None)

    cache = StarFeatureCache(
        cleaned,
        trend_windows=[indicator.trend_window],
        atr_windows=[indicator.atr_window],
        volatility_windows=[indicator.volatility_window],
        volume_windows=[indicator.volume_window],
        pattern_windows=[indicator.pattern_lookback],
    )

    raw_action, context, new_state = generate_realtime_signal(
        df=cleaned, 
        indicator_params=indicator, 
        model_params=model_params, 
        model=booster, 
        feature_columns=feature_columns, 
        class_means=class_means, 
        cache=cache, 
        state=runtime_state
    )
    colored_action = f"{YELLOW}{raw_action}{RESET}"
    context.update({"strategy": strategy, "symbol": symbol, "timeframe": timeframe})
    LOGGER.info("star_xgb action=%s details=%s", colored_action, context)

    if raw_action != "HOLD":
        dispatch_signal(raw_action, context)

    if new_state != runtime_state:
        save_runtime_state(state_store_path, strategy, symbol, timeframe, state=new_state.to_dict())
    
    return {"action": raw_action, "context": context, "state": new_state.to_dict()}


def main() -> None:
    parser = argparse.ArgumentParser(description="star_xgb realtime scheduler")
    parser.add_argument("--symbol", type=str, default="BTC/USDT")
    parser.add_argument("--timeframe", type=str, default="5m")
    parser.add_argument("--strategy", type=str, default="star_xgb_default", help="策略名稱 (同 study name)")
    parser.add_argument("--lookback-days", type=int, default=30) # 即時訊號不需要太長的歷史
    parser.add_argument("--interval-minutes", type=int, default=5)
    parser.add_argument("--params-db", type=Path, default=DEFAULT_STATE_DB)
    parser.add_argument("--state-db", type=Path, default=DEFAULT_STATE_DB)
    parser.add_argument("--exchange", type=str, default="binance")
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    args = parser.parse_args()

    _configure_logging(args.log_path)
    scheduler = BlockingScheduler()

    def _job_wrapper() -> None:
        try:
            run_star_cycle(
                symbol=args.symbol,
                timeframe=args.timeframe,
                strategy=args.strategy,
                lookback_days=args.lookback_days,
                params_store_path=args.params_db,
                state_store_path=args.state_db,
                exchange=args.exchange,
            )
        except Exception as exc:
            LOGGER.exception("star_xgb realtime cycle failed: %s", exc)

    scheduler.add_job(_job_wrapper, "interval", minutes=args.interval_minutes, next_run_time=pd.Timestamp.utcnow())
    LOGGER.info("Starting star_xgb scheduler | strategy=%s symbol=%s timeframe=%s interval=%sm", args.strategy, args.symbol, args.timeframe, args.interval_minutes)
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        LOGGER.info("Scheduler stopped")


if __name__ == "__main__":
    main()
