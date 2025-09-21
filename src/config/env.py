import os
from functools import lru_cache
from pathlib import Path
from typing import Dict

DEFAULT_ENV_PATH = Path("config/.env")


def _parse_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if "=" not in line:
        return None
    key, value = line.split("=", 1)
    return key.strip(), value.strip()


@lru_cache(maxsize=1)
def load_env(path: Path = DEFAULT_ENV_PATH) -> Dict[str, str]:
    """載入 `.env` 檔案並回傳 key-value。"""
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


__all__ = ["load_env", "DEFAULT_ENV_PATH"]
