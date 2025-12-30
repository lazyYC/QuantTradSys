from pathlib import Path
import os

# Project Roots
# config is in src/config, so:
SRC_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = SRC_DIR.parent

# Storage Paths
DEFAULT_STORAGE_DIR = ROOT_DIR / "storage"
# DEFAULT_STATE_DB = DEFAULT_STORAGE_DIR / "strategy_state.db" # Deprecated (Moved to Postgres)
# DEFAULT_MARKET_DB = DEFAULT_STORAGE_DIR / "market_data.db" # Deprecated (Moved to Postgres)
DEFAULT_LOG_DIR = DEFAULT_STORAGE_DIR / "logs"
