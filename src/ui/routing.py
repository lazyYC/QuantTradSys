"""UI 專用的 FastAPI 路由。"""
from __future__ import annotations

from datetime import timedelta
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from ui.data_service import (
    CandleQuery,
    TradeQuery,
    fetch_candles,
    fetch_metrics,
    fetch_trades,
    list_strategy_configs,
    parse_timestamp,
)

router = APIRouter(prefix="/api", tags=["ui"])


@router.get("/configs")
def get_configs() -> list[dict[str, str]]:
    """提供策略 / 交易對 / timeframe 選項列表。"""

    return list_strategy_configs()


@router.get("/candles")
def get_candles(
    symbol: str,
    timeframe: str = Query("5m", description="資料庫內的原始 timeframe"),
    start: Optional[str] = Query(None, description="ISO8601 起始時間"),
    end: Optional[str] = Query(None, description="ISO8601 結束時間"),
    target_timeframe: Optional[str] = Query(None, description="顯示用 timeframe，需為原始 timeframe 的整數倍"),
) -> dict:
    start_ts = parse_timestamp(start)
    end_ts = parse_timestamp(end)
    if start_ts and end_ts and start_ts >= end_ts:
        raise HTTPException(status_code=400, detail="start must be earlier than end")
    query = CandleQuery(
        symbol=symbol,
        base_timeframe=timeframe,
        start=start_ts,
        end=end_ts,
        target_timeframe=target_timeframe,
    )
    try:
        candles = fetch_candles(query)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    records = candles.copy()
    if not records.empty:
        records["timestamp"] = records["timestamp"].dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return {"candles": records.to_dict(orient="records")}


@router.get("/trades")
def get_trades(
    strategy: str,
    symbol: str,
    timeframe: str,
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
    dataset: Optional[str] = Query(None, description="train/test 或其他標籤"),
) -> dict:
    start_ts = parse_timestamp(start)
    end_ts = parse_timestamp(end)
    query = TradeQuery(
        strategy=strategy,
        symbol=symbol,
        timeframe=timeframe,
        start=start_ts,
        end=end_ts,
        dataset=dataset,
    )
    trades = fetch_trades(query)
    if not trades.empty:
        trades["entry_time"] = trades["entry_time"].dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        trades["exit_time"] = trades["exit_time"].dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return {"trades": trades.to_dict(orient="records")}


@router.get("/summary")
def get_summary(
    strategy: str,
    symbol: str,
    timeframe: str,
) -> dict:
    metrics = fetch_metrics(strategy, symbol, timeframe)
    if metrics.empty:
        return {"metrics": []}
    df = metrics.copy()
    df["created_at"] = df["created_at"].dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return {"metrics": df.to_dict(orient="records")}


__all__ = ["router"]
