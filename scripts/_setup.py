"""
Common setup module for scripts.
Handles path setup, configuration loading, and common constants.
"""
import sys
import os
import logging
from pathlib import Path

# 1. Setup Path
# Assumes this file is in scripts/ directory, so project root is one level up
SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPTS_DIR.parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# 2. Load .env from Project Root
# Try to use python-dotenv if available
try:
    from dotenv import load_dotenv
    env_path = ROOT_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        # logging.getLogger(__name__).info(f"Loaded .env from {env_path}")
except ImportError:
    # logging.getLogger(__name__).warning("python-dotenv not installed, skipping .env loading")
    pass

# 3. Common Constants
DEFAULT_STATE_DB = ROOT_DIR / "storage" / "strategy_state.db"
DEFAULT_MARKET_DB = ROOT_DIR / "storage" / "market_data.db"
DEFAULT_LOG_DIR = ROOT_DIR / "storage" / "logs"

def setup_logging(log_path: Path | None = None, level: int = logging.INFO) -> None:
    """Configure logging with a common format."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        )
        handlers.append(file_handler)
        
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True
    )
