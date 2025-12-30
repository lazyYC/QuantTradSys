import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import select

from database.connection import get_session
from database.schema import StrategyParam

LOGGER = logging.getLogger(__name__)

@dataclass
class StrategyRecord:
    strategy: str
    study: str
    symbol: str
    timeframe: str
    params: Dict[str, Any]
    metrics: Dict[str, Any]
    model_path: Optional[str]
    updated_at: str


def save_strategy_params(
    db_path: Path, # Unused
    strategy: str,
    study: str,
    symbol: str,
    timeframe: str,
    params: Dict[str, Any],
    metrics: Dict[str, Any],
    model_path: Optional[str] = None,
    *,
    stop_loss_pct: float = 0.005,
    transaction_cost: float = 0.001,
) -> StrategyRecord:
    """Persist strategy params to Postgres."""
    now = datetime.now(timezone.utc)
    
    # Merge extra config into params for storage (legacy behavior)
    payload = dict(params)
    payload["stop_loss_pct"] = stop_loss_pct
    payload["transaction_cost"] = transaction_cost
    
    with get_session() as session:
        stmt = insert(StrategyParam).values(
            strategy=strategy,
            study=study,
            symbol=symbol,
            timeframe=timeframe,
            params_data=payload,
            metrics_data=metrics,
            model_path=model_path,
            updated_at=now
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["strategy", "study", "symbol", "timeframe"],
            set_={
                "params_data": stmt.excluded.params_data,
                "metrics_data": stmt.excluded.metrics_data,
                "model_path": stmt.excluded.model_path,
                "updated_at": stmt.excluded.updated_at
            }
        )
        session.execute(stmt)

    LOGGER.info("Stored parameters for %s | %s | %s | %s", strategy, study, symbol, timeframe)
    return StrategyRecord(
        strategy=strategy,
        study=study,
        symbol=symbol,
        timeframe=timeframe,
        params=payload,
        metrics=metrics,
        model_path=model_path,
        updated_at=now.isoformat()
    )


def load_strategy_params(
    db_path: Path, # Unused
    strategy: str,
    study: str,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
) -> Optional[StrategyRecord]:
    """Load latest strategy params."""
    with get_session() as session:
        stmt = select(StrategyParam).where(
            StrategyParam.strategy == strategy,
            StrategyParam.study == study
        )
        
        if symbol:
            stmt = stmt.where(StrategyParam.symbol == symbol)
        if timeframe:
            stmt = stmt.where(StrategyParam.timeframe == timeframe)
            
        # If multiple matches (implicit wildcards not fully supported by PK structure but legacy code implied it),
        # Order by updated_at desc
        stmt = stmt.order_by(StrategyParam.updated_at.desc())
        
        result = session.execute(stmt).scalars().first()
        
        if result is None:
            return None
            
        return StrategyRecord(
            strategy=result.strategy,
            study=result.study,
            symbol=result.symbol,
            timeframe=result.timeframe,
            params=result.params_data,
            metrics=result.metrics_data,
            model_path=result.model_path,
            updated_at=result.updated_at.isoformat() if result.updated_at else ""
        )

__all__ = [
    "StrategyRecord",
    "save_strategy_params",
    "load_strategy_params",
]
