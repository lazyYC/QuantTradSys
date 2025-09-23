"""常用的數值格式化與 round 工具。"""
from numbers import Integral, Real
from typing import Any, Iterable, Mapping, Optional

# 預設當作整數處理的績效欄位名稱
METRIC_INT_FIELDS = frozenset({"trades"})


def round_numeric_fields(
    data: Mapping[str, Any],
    *,
    decimals_map: Optional[Mapping[str, int]] = None,
    default_decimals: int = 6,
    int_fields: Optional[Iterable[str]] = None,
) -> dict:
    """對映射中的數值欄位進行 round，避免浮點尾數過長。"""
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


def format_metrics(
    metrics: Mapping[str, Any],
    *,
    decimals: int = 6,
    int_fields: Iterable[str] = METRIC_INT_FIELDS,
) -> dict:
    """統一格式化策略績效指標。"""
    return round_numeric_fields(metrics, default_decimals=decimals, int_fields=int_fields)


def format_params(
    params: Mapping[str, Any],
    *,
    decimals_map: Mapping[str, int],
    default_decimals: int = 6,
    int_fields: Optional[Iterable[str]] = None,
) -> dict:
    """依指定小數位數對策略參數進行 round。"""
    return round_numeric_fields(
        params,
        decimals_map=decimals_map,
        default_decimals=default_decimals,
        int_fields=int_fields,
    )


__all__ = [
    "METRIC_INT_FIELDS",
    "round_numeric_fields",
    "format_metrics",
    "format_params",
]
