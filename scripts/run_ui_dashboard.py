"""啟動 UI Dashboard 服務。"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ui.server import create_app  # noqa: E402

LOGGER = logging.getLogger(__name__)


def _configure_logging() -> None:
    """設定基礎 logging，避免重複初始化。"""

    if logging.getLogger().handlers:
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run UI dashboard server")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=7861, help="Server port")
    parser.add_argument(
        "--api-only",
        action="store_true",
        help="只啟動 FastAPI，不掛載 Gradio 介面",
    )
    args = parser.parse_args()

    _configure_logging()
    app = create_app(with_gradio=not args.api_only)
    LOGGER.info(
        "Starting dashboard server | host=%s port=%s api_only=%s",
        args.host,
        args.port,
        args.api_only,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
