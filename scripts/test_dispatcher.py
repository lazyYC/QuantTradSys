#!/usr/bin/env python
"""Quick dispatcher sanity tests with mocked external services."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest import mock, TestCase, main


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import notifier.dispatcher as dispatcher  # noqa: E402


class DispatcherTests(TestCase):
    def setUp(self) -> None:
        dispatcher._ALPACA_CLIENT = None  # reset cached client
        os.environ.setdefault("ALPACA_API_KEY", "test_key")
        os.environ.setdefault("ALPACA_SECRET", "test_secret")
        os.environ.setdefault("DISCORD_WEBHOOK", "https://example.com/hook")

    @mock.patch("notifier.dispatcher.requests.post")
    @mock.patch("notifier.dispatcher.execute_trading")
    def test_dispatch_signal_invokes_trading(
        self, mock_execute: mock.MagicMock, mock_post: mock.MagicMock
    ) -> None:
        dispatcher.dispatch_signal(
            "ENTER_LONG",
            {"symbol": "BTC/USDT", "price": 30000.0},
            env_path=SRC_DIR / "config" / ".env",
        )
        mock_execute.assert_called_once()
        mock_post.assert_called_once()

    def test_normalize_symbol_variants(self) -> None:
        self.assertEqual(dispatcher._normalize_symbol("BTC-USDT"), "BTC/USD")
        self.assertEqual(dispatcher._normalize_symbol("btcusd"), "BTC/USD")
        self.assertEqual(dispatcher._normalize_symbol(None), "BTC/USD")


if __name__ == "__main__":
    main()
