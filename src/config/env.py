"""環境變數載入工具。"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

ENV_OVERRIDE_KEY = "QTS_ENV_PATH"
MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parents[2]
DEFAULT_ENV_PATH = MODULE_DIR / ".env"

PathLike = Union[str, Path]


def load_env(path: Optional[PathLike] = None) -> Dict[str, str]:
    """讀取 `.env` 檔案，並將鍵值寫入環境變數。"""
    for candidate in _candidate_paths(path):
        if candidate.exists():
            return _load_env_internal(candidate)
    return {}


def _candidate_paths(custom: Optional[PathLike]) -> Iterable[Path]:
    """依優先順序產出可能的 `.env` 檔案路徑。"""
    env_override = os.getenv(ENV_OVERRIDE_KEY)
    raw_candidates: list[Optional[PathLike]] = [
        custom,
        env_override,
        DEFAULT_ENV_PATH,
        PROJECT_ROOT / "config/.env",
        PROJECT_ROOT / ".env",
    ]
    for candidate in raw_candidates:
        if candidate is None:
            continue
        candidate_path = Path(candidate)
        resolved = (
            candidate_path
            if candidate_path.is_absolute()
            else (Path.cwd() / candidate_path)
        )
        yield resolved.resolve()


def _parse_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if "=" not in line:
        return None
    key, value = line.split("=", 1)
    return key.strip(), value.strip()


@lru_cache(maxsize=16)
def _load_env_internal(path: Path) -> Dict[str, str]:
    """將 `.env` 檔案內容轉為字典並寫入環境變數。"""
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_line(line)
        if not parsed:
            continue
        key, value = parsed
        os.environ.setdefault(key, value)
        values[key] = value
    return values


__all__ = ["load_env", "DEFAULT_ENV_PATH", "ENV_OVERRIDE_KEY"]
