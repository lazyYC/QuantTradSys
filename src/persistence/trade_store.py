"""
Trade Store.
Encapsulates Strategy Trades and Metrics persistence using SQLAlchemy (PostgreSQL).
"""
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional

import pandas as pd
from sqlalchemy import select, delete
from sqlalchemy.dialects.postgresql import insert

from database.connection import get_session, get_engine
from database.schema import StrategyTrade, StrategyMetric

LOGGER = logging.getLogger(__name__)


def load_trades(
    strategy: str,
    study: Optional[str] = None,
    dataset: Optional[str] = None,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    run_id: Optional[str] = None,
) -> pd.DataFrame:
    """Read trades matching filters."""
    stmt = select(StrategyTrade).where(StrategyTrade.strategy == strategy)
    
    if study:
        stmt = stmt.where(StrategyTrade.study == study)
    if dataset:
        stmt = stmt.where(StrategyTrade.dataset == dataset)
    if symbol:
        stmt = stmt.where(StrategyTrade.symbol == symbol)
    if timeframe:
        stmt = stmt.where(StrategyTrade.timeframe == timeframe)
    if run_id:
        stmt = stmt.where(StrategyTrade.run_id == run_id)
        
    stmt = stmt.order_by(StrategyTrade.entry_time.asc())

    with get_engine().connect() as conn:
        df = pd.read_sql_query(stmt, conn)
        
    if df.empty:
        return pd.DataFrame()

    # Normalize Columns if needed? SQLAlchemy/Pandas usually handles datetimes well.
    # Just ensure return_pct maps to 'return' if legacy code expects it
    if "return_pct" in df.columns:
        df = df.rename(columns={"return_pct": "return"})

    return df


def load_metrics(
    strategy: str,
    study: Optional[str] = None,
    dataset: Optional[str] = None,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    run_id: Optional[str] = None,
) -> pd.DataFrame:
    """Read metrics matching filters."""
    stmt = select(StrategyMetric).where(StrategyMetric.strategy == strategy)
    
    if study:
        stmt = stmt.where(StrategyMetric.study == study)
    if dataset:
        stmt = stmt.where(StrategyMetric.dataset == dataset)
    if symbol:
        stmt = stmt.where(StrategyMetric.symbol == symbol)
    if timeframe:
        stmt = stmt.where(StrategyMetric.timeframe == timeframe)
    if run_id:
        stmt = stmt.where(StrategyMetric.run_id == run_id)
        
    stmt = stmt.order_by(StrategyMetric.created_at.desc())

    with get_engine().connect() as conn:
        df = pd.read_sql_query(stmt, conn)
        
    return df


def save_trades(
    *,
    strategy: str,
    study: str,
    dataset: str,
    symbol: str,
    timeframe: str,
    trades: pd.DataFrame,
    metrics: Mapping[str, float],
    run_id: Optional[str] = None,
) -> str:
    """Persist trades and metrics."""
    run_identifier = run_id or datetime.now(timezone.utc).isoformat()
    created_at = datetime.now(timezone.utc)
    
    with get_session() as session:
        # 1. Upsert Trades
        if not trades.empty:
            trade_records = []
            LOGGER.info(f"Saving {len(trades)} trades for {strategy} | {study} ({dataset})...")
            for _, row in trades.iterrows():
                # Handle potentially missing or differently named cols from backtest
                 entry_time = pd.to_datetime(row["entry_time"], utc=True)
                 exit_time = pd.to_datetime(row["exit_time"], utc=True)
                 
                 trade_records.append({
                     "run_id": run_identifier,
                     "strategy": strategy,
                     "study": study,
                     "dataset": dataset,
                     "symbol": symbol,
                     "timeframe": timeframe,
                     "entry_time": entry_time,
                     "exit_time": exit_time,
                     "side": str(row["side"]),
                     "entry_price": float(row["entry_price"]),
                     "exit_price": float(row["exit_price"]),
                     "return_pct": float(row["return"]), # Map 'return' -> 'return_pct'
                     "holding_mins": float(row["holding_mins"]),
                     "entry_zscore": float(row.get("entry_zscore", 0.0)),
                     "exit_zscore": float(row.get("exit_zscore", 0.0)),
                     "exit_reason": str(row.get("exit_reason", "")),
                 })
            
            stmt = insert(StrategyTrade).values(trade_records)
            stmt = stmt.on_conflict_do_update(
                index_elements=["run_id", "dataset", "entry_time"],
                set_={
                    "exit_time": stmt.excluded.exit_time,
                    "exit_price": stmt.excluded.exit_price,
                    "return_pct": stmt.excluded.return_pct,
                    "exit_reason": stmt.excluded.exit_reason
                }
            )
            session.execute(stmt)

        # 2. Upsert Metrics
        metric_record = {
            "run_id": run_identifier,
            "strategy": strategy,
            "study": study,
            "dataset": dataset,
            "symbol": symbol,
            "timeframe": timeframe,
            "annualized_return": float(metrics.get("annualized_return", 0.0)),
            "total_return": float(metrics.get("total_return", 0.0)),
            "sharpe": float(metrics.get("sharpe", 0.0)),
            "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
            "win_rate": float(metrics.get("win_rate", 0.0)),
            "trades_count": int(metrics.get("trades", 0)),
            "period_start": pd.to_datetime(metrics.get("period_start"), utc=True) if metrics.get("period_start") else None,
            "period_end": pd.to_datetime(metrics.get("period_end"), utc=True) if metrics.get("period_end") else None,
            "created_at": created_at
        }
        
        stmt_m = insert(StrategyMetric).values([metric_record])
        stmt_m = stmt_m.on_conflict_do_update(
            index_elements=["run_id", "dataset"],
            set_={
                "total_return": stmt_m.excluded.total_return,
                "sharpe": stmt_m.excluded.sharpe,
                "trades_count": stmt_m.excluded.trades_count,
                "created_at": stmt_m.excluded.created_at
            }
        )
        session.execute(stmt_m)

    LOGGER.info(
        "Stored %s trades for %s | %s (%s) run=%s (Postgres)",
        len(trades), strategy, study, dataset, run_identifier
    )
    return run_identifier


def prune_strategy_trades(
    *,
    strategy: str,
    study: str,
    symbol: str,
    timeframe: str,
    keep_run_id: str,
) -> int:
    """Delete old run trades, keeping specific run_id."""
    with get_session() as session:
        stmt = delete(StrategyTrade).where(
            StrategyTrade.strategy == strategy,
            StrategyTrade.study == study,
            StrategyTrade.symbol == symbol,
            StrategyTrade.timeframe == timeframe,
            StrategyTrade.run_id != keep_run_id
        )
        result = session.execute(stmt)
        return result.rowcount


def prune_strategy_metrics(
    *,
    strategy: str,
    study: str,
    symbol: str,
    timeframe: str,
    keep_run_id: str,
) -> int:
    """Delete old run metrics, keeping specific run_id."""
    with get_session() as session:
        stmt = delete(StrategyMetric).where(
            StrategyMetric.strategy == strategy,
            StrategyMetric.study == study,
            StrategyMetric.symbol == symbol,
            StrategyMetric.timeframe == timeframe,
            StrategyMetric.run_id != keep_run_id
        )
        result = session.execute(stmt)
        return result.rowcount

__all__ = [
    "save_trades",
    "load_trades",
    "load_metrics",
    "prune_strategy_trades",
    "prune_strategy_metrics",
]
