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


def _parse_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if "=" not in line:
        return None
    key, value = line.split("=", 1)
    return key.strip(), value.strip()


def _candidate_paths(custom: Optional[PathLike]) -> Iterable[Path]:
    """回傳依優先順序排序的候選路徑。"""

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
        resolved = candidate_path if candidate_path.is_absolute() else (Path.cwd() / candidate_path)
        yield resolved.resolve()


@lru_cache(maxsize=16)
def _load_env_internal(path: Path) -> Dict[str, str]:
    """實際載入 `.env` 檔案並回傳 key-value。"""

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


def load_env(path: Optional[PathLike] = None) -> Dict[str, str]:
    """載入 `.env` 檔案，優先順序：
    1. 函式參數提供的路徑。
    2. `QTS_ENV_PATH` 環境變數指向的檔案。
    3. `src/config/.env`（模組所在目錄）。
    4. 專案根目錄下的 `config/.env` 或 `.env`。
    """

    for candidate in _candidate_paths(path):
        if candidate.exists():
            return _load_env_internal(candidate)
    return {}


__all__ = ["load_env", "DEFAULT_ENV_PATH", "ENV_OVERRIDE_KEY"]
