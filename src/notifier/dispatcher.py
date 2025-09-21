import logging
from typing import Any, Dict

LOGGER = logging.getLogger(__name__)


def dispatch_signal(action: str, context: Dict[str, Any]) -> None:
    """將策略訊號發佈到外部通道，目前以日誌方式模擬。"""
    LOGGER.info("Dispatch signal=%s context=%s", action, context)


__all__ = ["dispatch_signal"]
