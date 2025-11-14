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
        mock_execute.return_value = (True, "ok")
        result = dispatcher.dispatch_signal(
            "ENTER_LONG",
            {"symbol": "BTC/USDT:USDT", "price": 30000.0},
            env_path=SRC_DIR / "config" / ".env",
        )
        mock_execute.assert_called_once()
        self.assertEqual(mock_post.call_count, 2)
        self.assertTrue(result)

    def test_normalize_symbol_variants(self) -> None:
        self.assertEqual(dispatcher._normalize_symbol("BTC-USDT"), "BTC/USDT")
        self.assertEqual(dispatcher._normalize_symbol("btcusd"), "BTC/USD")
        self.assertEqual(dispatcher._normalize_symbol(None), "BTC/USDT:USDT")
        self.assertEqual(dispatcher.format_alpaca_symbol("BTC/USDT"), "BTCUSD")


if __name__ == "__main__":
    main()
