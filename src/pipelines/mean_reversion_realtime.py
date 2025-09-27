import logging
from pathlib import Path
from typing import Optional

from data_pipeline.ccxt_fetcher import fetch_yearly_ohlcv
from notifier.dispatcher import dispatch_signal
from persistence.param_store import load_strategy_params, StrategyRecord
from persistence.runtime_store import load_runtime_state, save_runtime_state
from strategies.data_utils import prepare_ohlcv_frame
from strategies.mean_reversion import (
    MeanReversionParams,
    MeanReversionRuntimeState,
    generate_realtime_decision,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_STATE_DB = Path("storage/strategy_state.db")


def _load_params(record: StrategyRecord | None) -> Optional[MeanReversionParams]:
    if record is None:
        return None
    try:
        return MeanReversionParams(**record.params)
    except TypeError as exc:  # noqa: BLE001
        LOGGER.error("Parameter record malformed: params=%s", record.params)
        raise exc


def run_realtime_cycle(
    symbol: str,
    *,
    strategy: str = "mean_reversion_optuna",
    timeframe: str = "5m",
    lookback_days: int = 400,
    params_store_path: Path = DEFAULT_STATE_DB,
    state_store_path: Path = DEFAULT_STATE_DB,
    exchange_id: str = "binance",
    exchange_config: Optional[dict] = None,
) -> dict:
    """抓取最新資料、計算訊號並視需要派送通知。"""
    param_record = load_strategy_params(
        params_store_path,
        strategy=strategy,
        symbol=symbol,
        timeframe=timeframe,
    )
    params = _load_params(param_record)
    if params is None:
        LOGGER.warning("No mean reversion parameters stored for %s %s", symbol, timeframe)
        return {"action": "HOLD", "reason": "missing_params"}

    raw_df = fetch_yearly_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        exchange_id=exchange_id,
        exchange_config=exchange_config,
        lookback_days=lookback_days,
    )
    cleaned = prepare_ohlcv_frame(raw_df, timeframe)
    if cleaned.empty:
        LOGGER.warning("No OHLCV data available for %s %s", symbol, timeframe)
        return {"action": "HOLD", "reason": "empty_data"}

    runtime_record = load_runtime_state(
        state_store_path,
        strategy=strategy,
        symbol=symbol,
        timeframe=timeframe,
    )
    runtime_state = MeanReversionRuntimeState.from_dict(runtime_record.state if runtime_record else None)

    action, context, new_state = generate_realtime_decision(cleaned, params, state=runtime_state)
    context.update({"strategy": strategy, "symbol": symbol, "timeframe": timeframe})
    LOGGER.info("Realtime action=%s context=%s", action, context)

    if action != "HOLD":
        dispatch_signal(action, context)

    if new_state != runtime_state:
        save_runtime_state(
            state_store_path,
            strategy=strategy,
            symbol=symbol,
            timeframe=timeframe,
            state=new_state.to_dict(),
        )

    return {"action": action, "context": context, "state": new_state.to_dict()}


__all__ = ["run_realtime_cycle", "DEFAULT_STATE_DB"]
