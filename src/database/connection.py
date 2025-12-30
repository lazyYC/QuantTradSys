import logging
import os
import time
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker, scoped_session
from sqlalchemy.exc import OperationalError

from config.env import load_env

LOGGER = logging.getLogger(__name__)

# Global engine/session instances
_ENGINE: Engine | None = None
_SESSION_FACTORY: sessionmaker | None = None


def get_database_url() -> str:
    """Retrieve DATABASE_URL from environment."""
    load_env()
    url = os.getenv("DATABASE_URL")
    if not url:
        raise ValueError("DATABASE_URL environment variable is not set.")
    return url


def get_engine() -> Engine:
    """Get or create the SQLAlchemy engine singleton."""
    global _ENGINE
    if _ENGINE is None:
        url = get_database_url()
        # Recommended settings for production resource handling
        _ENGINE = create_engine(
            url,
            pool_pre_ping=True,  # Auto-reconnect on stale connection
            pool_size=10,        # Default pool size
            max_overflow=20,     # Max extra connections
            connect_args={"connect_timeout": 10} # Optional: timeouts
        )
        LOGGER.info("Database engine initialized.")
    return _ENGINE


def test_connection(retries: int = 3, delay: int = 2) -> bool:
    """
    Test database connectivity with retries.
    Useful for waking up serverless DBs (Supabase) or handling transient network issues.
    """
    engine = get_engine()
    for attempt in range(1, retries + 1):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            LOGGER.info("Database connection verified.")
            return True
        except OperationalError as e:
            LOGGER.warning(f"Connection attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                time.sleep(delay)
            else:
                LOGGER.error("All connection attempts failed.")
                return False
    return False


def get_session_factory() -> sessionmaker:
    """Get the session factory."""
    global _SESSION_FACTORY
    if _SESSION_FACTORY is None:
        engine = get_engine()
        _SESSION_FACTORY = sessionmaker(bind=engine, autoflush=False)
    return _SESSION_FACTORY


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Provide a transactional scope around a series of operations.
    Usage:
        with get_session() as session:
            session.add(...)
    """
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def initialize_db_schema(clean: bool = False):
    """Create all tables defined in metadata."""
    from database.schema import Base
    
    if not test_connection():
        raise ConnectionError("Could not connect to database. Check your network or DATABASE_URL.")

    engine = get_engine()
    
    if clean:
        Base.metadata.drop_all(engine)
        LOGGER.info("Dropped all tables (clean=True).")

    Base.metadata.create_all(engine)
    LOGGER.info("Database schema initialized (create_all).")
