import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional

def setup_logging(log_path: Optional[Path] = None, level: int = logging.INFO) -> None:
    """Configure logging with a common format."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # 2MB max size, 3 backups
        file_handler = RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        handlers.append(file_handler)
        
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True
    )
