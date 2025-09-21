import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from config.env import DEFAULT_ENV_PATH, load_env

LOGGER = logging.getLogger(__name__)


def _send_discord(webhook: str, action: str, context: Dict[str, Any]) -> None:
    message = context.get("message") or f"Signal: {action}\nContext: {context}"
    payload = {"content": message}
    try:
        response = requests.post(webhook, json=payload, timeout=10)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Failed to dispatch signal to Discord: %s", exc)
    else:
        LOGGER.info("Sent signal to Discord webhook")


def dispatch_signal(
    action: str,
    context: Dict[str, Any],
    *,
    env_path: Optional[Path] = None,
) -> None:
    """將策略訊號發佈到外部通道。"""
    LOGGER.info("Dispatch signal=%s context=%s", action, context)
    load_env(env_path or DEFAULT_ENV_PATH)
    if action.upper() == "HOLD":
        return
    webhook = os.getenv("DISCORD_WEBHOOK")
    if not webhook:
        LOGGER.debug("DISCORD_WEBHOOK not configured, skipping Discord notification")
        return
    _send_discord(webhook, action, context)


__all__ = ["dispatch_signal"]
