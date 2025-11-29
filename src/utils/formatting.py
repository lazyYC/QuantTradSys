"""常用的數值格式化與 round 工具函式。"""

from numbers import Integral, Real
from typing import Any, Iterable, Mapping, Optional

# 預設整數型的績效欄位
METRIC_INT_FIELDS = frozenset({"trades"})


def format_metrics(
    metrics: Mapping[str, Any],
    *,
    decimals: int = 6,
    int_fields: Iterable[str] = METRIC_INT_FIELDS,
) -> dict:
    """統一回測輸出的績效指標精度。"""
    return round_numeric_fields(
        metrics, default_decimals=decimals, int_fields=int_fields
    )


def format_params(
    params: Mapping[str, Any],
    *,
    decimals_map: Mapping[str, int],
    default_decimals: int = 6,
    int_fields: Optional[Iterable[str]] = None,
) -> dict:
    """依據指定小數位配置策略參數的 round 規則。"""
    return round_numeric_fields(
        params,
        decimals_map=decimals_map,
        default_decimals=default_decimals,
        int_fields=int_fields,
    )


def round_numeric_fields(
    data: Mapping[str, Any],
    *,
    decimals_map: Optional[Mapping[str, int]] = None,
    default_decimals: int = 6,
    int_fields: Optional[Iterable[str]] = None,
) -> dict:
    """對映射中的數值欄位進行 round，避免浮點尾巴過長。"""
    decimals_map = decimals_map or {}
    int_fields_set = set(int_fields or [])
    rounded: dict = {}
    for key, value in data.items():
        if value is None:
            rounded[key] = None
            continue
        if isinstance(value, bool):
            rounded[key] = value
            continue
        if key in int_fields_set:
            rounded[key] = int(round(float(value)))
            continue
        if key in decimals_map:
            rounded[key] = round(float(value), decimals_map[key])
            continue
        if isinstance(value, Integral):
            rounded[key] = int(value)
            continue
        if isinstance(value, Real):
            rounded[key] = round(float(value), default_decimals)
            continue
        rounded[key] = value
    return rounded


def _format_ts(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()


__all__ = [
    "METRIC_INT_FIELDS",
    "format_metrics",
    "format_params",
    "round_numeric_fields",
    "_format_ts",
]
