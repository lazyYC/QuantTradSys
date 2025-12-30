import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import select

from database.connection import get_session, get_engine
from database.schema import RuntimeState

LOGGER = logging.getLogger(__name__)

@dataclass
class RuntimeRecord:
    strategy: str
    study: str
    symbol: str
    timeframe: str
    state: Dict[str, Any]
    updated_at: str


def save_runtime_state(
    db_path: Path, # Kept for compatibility but unused
    *,
    strategy: str,
    study: str,
    symbol: str,
    timeframe: str,
    state: Dict[str, Any],
) -> RuntimeRecord:
    """Persist runtime state to Postgres."""
    # payload = json.dumps(state) # SQLAlchemy JSON type handles dict automatically
    now = datetime.now(timezone.utc)
    
    with get_session() as session:
        stmt = insert(RuntimeState).values(
            strategy=strategy,
            study=study,
            symbol=symbol,
            timeframe=timeframe,
            state_data=state,
            updated_at=now
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["strategy", "study", "symbol", "timeframe"],
            set_={
                "state_data": stmt.excluded.state_data,
                "updated_at": stmt.excluded.updated_at
            }
        )
        session.execute(stmt)

    LOGGER.debug(
        "Saved runtime state strategy=%s study=%s symbol=%s timeframe=%s",
        strategy, study, symbol, timeframe
    )
    
    return RuntimeRecord(
        strategy=strategy,
        study=study,
        symbol=symbol,
        timeframe=timeframe,
        state=state,
        updated_at=now.isoformat()
    )


def load_runtime_state(
    db_path: Path, # Unused
    strategy: str,
    study: str,
    symbol: str,
    timeframe: str,
) -> Optional[RuntimeRecord]:
    """Load persisted runtime state."""
    with get_session() as session:
        stmt = select(RuntimeState).where(
            RuntimeState.strategy == strategy,
            RuntimeState.study == study,
            RuntimeState.symbol == symbol,
            RuntimeState.timeframe == timeframe
        )
        result = session.execute(stmt).scalar_one_or_none()
        
        if result is None:
            return None
            
        return RuntimeRecord(
            strategy=result.strategy,
            study=result.study,
            symbol=result.symbol,
            timeframe=result.timeframe,
            state=result.state_data, # JSON dict
            updated_at=result.updated_at.isoformat() if result.updated_at else ""
        )

__all__ = [
    "RuntimeRecord",
    "save_runtime_state",
    "load_runtime_state",
]
