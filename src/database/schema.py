"""
SQLAlchemy Schema Definitions for QuantTradSys.
Compatible with PostgreSQL (Supabase).
"""
import datetime
from sqlalchemy import (
    Column,
    String,
    Float,
    BigInteger,
    DateTime,
    Integer,
    UniqueConstraint,
    Index,
    JSON
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class OHLCV(Base):
    """
    Market Data (K-Lines).
    Composite Key: (symbol, timeframe, ts)
    """
    __tablename__ = "ohlcv"

    symbol = Column(String, primary_key=True)
    timeframe = Column(String, primary_key=True)
    ts = Column(BigInteger, primary_key=True)  # Unix Timestamp (ms)
    
    # Redundant but useful timestamp for Postgres queries/partitions
    iso_ts = Column(DateTime(timezone=True), index=True) 
    
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)

    __table_args__ = (
        Index("idx_ohlcv_symbol_time", "symbol", "timeframe", "ts"),
    )


class StrategyTrade(Base):
    """
    Execution records for strategies.
    Key: (run_id, dataset, entry_time)
    """
    __tablename__ = "strategy_trades"

    run_id = Column(String, primary_key=True)
    strategy = Column(String, nullable=False)
    study = Column(String, nullable=False)
    dataset = Column(String, primary_key=True) # backtest, valid, train, realtime
    symbol = Column(String, nullable=False)
    timeframe = Column(String, nullable=False)
    
    entry_time = Column(DateTime(timezone=True), primary_key=True)
    exit_time = Column(DateTime(timezone=True))
    
    side = Column(String, nullable=False) # LONG / SHORT
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=False)
    return_pct = Column(Float, nullable=False)
    
    holding_mins = Column(Float)
    
    # Store extended attributes (zscores etc) in JSON to be flexible? 
    # Or keep strict columns as previous version? 
    # Previous: entry_zscore, exit_zscore, exit_reason were explicit columns.
    entry_zscore = Column(Float)
    exit_zscore = Column(Float)
    exit_reason = Column(String)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("idx_trades_lookup", "strategy", "study", "symbol"),
    )


class StrategyMetric(Base):
    """
    Performance summary for a strategy run/dataset.
    Key: (run_id, dataset)
    """
    __tablename__ = "strategy_metrics"

    run_id = Column(String, primary_key=True)
    dataset = Column(String, primary_key=True)
    
    strategy = Column(String, nullable=False)
    study = Column(String, nullable=False)
    symbol = Column(String, nullable=False)
    timeframe = Column(String, nullable=False)
    
    annualized_return = Column(Float)
    total_return = Column(Float)
    sharpe = Column(Float)
    max_drawdown = Column(Float)
    win_rate = Column(Float)
    trades_count = Column(Integer)
    
    period_start = Column(DateTime(timezone=True))
    period_end = Column(DateTime(timezone=True))
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("idx_metrics_lookup", "strategy", "study", "symbol"),
    )


class RuntimeState(Base):
    """
    Persisted runtime state for realtime strategies.
    Replaces the key-value usage in strategy_state.db
    """
    __tablename__ = "runtime_states"

    strategy = Column(String, primary_key=True)
    study = Column(String, primary_key=True)
    symbol = Column(String, primary_key=True)
    timeframe = Column(String, primary_key=True)
    
    # JSONB or JSON column for the state dictionary
    state_data = Column(JSON, nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class StrategyParam(Base):
    """
    Best parameters and metrics for a strategy/study.
    Key: (strategy, study, symbol, timeframe)
    """
    __tablename__ = "strategy_params"

    strategy = Column(String, primary_key=True)
    study = Column(String, primary_key=True)
    symbol = Column(String, primary_key=True)
    timeframe = Column(String, primary_key=True)
    
    params_data = Column(JSON, nullable=False)
    metrics_data = Column(JSON, nullable=False)
    model_path = Column(String, nullable=True)
    
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
