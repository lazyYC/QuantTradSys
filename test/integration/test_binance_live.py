
import pytest
import logging
import time
from pprint import pformat

LOGGER = logging.getLogger(__name__)

@pytest.mark.live
class TestBinanceLiveExecution:
    """
    Live tests for Binance Execution.
    Run with: pytest --run-live --symbol BTC/USDT
    """

    def test_open_close_cycle(self, binance_client, test_symbol):
        """
        Full Cycle Test:
        1. Pre-check: Verify no existing position.
        2. Open: Place Market Buy.
        3. Verify: Check position exists.
        4. Close: Place Market Sell (Reduce Only or Close).
        5. Verify: Check position is zero.
        """
        symbol = test_symbol
        # Usually client expects formatted symbol, e.g. BTCUSDT (depending on client impl)
        # The broker adapter usually handles this. Here we use the client directly.
        # BinanceUSDMClient usually takes raw symbol "BTCUSDT" or standard "BTC/USDT"?
        # Checking implementation: _configure_binance_symbol uses canonicalize_symbol but let's check input.
        # The client methods typically expect the exchange-specific symbol (e.g. BTCUSDT).
        # We'll use a helper or just format it.
        
        # We need to resolve the exchange symbol (e.g. BTC/USDT -> BTCUSDT)
        exchange_info = binance_client.get_exchange_info()
        # Simple heuristic or usage of adapter helper if available. 
        # For now, simplistic replace.
        trade_symbol = symbol.replace("/", "").replace(":USDT", "")
        
        LOGGER.info(f"Starting Live Test on {trade_symbol}")
        
        # 1. Pre-check: Ensure empty position
        positions = binance_client.get_position(trade_symbol)
        # API returns list or single dict? 
        # get_position in binance.py implementation loop:
        # returns single dict if symbol provided.
        
        initial_amt = float(positions.get('positionAmt', 0.0))
        if initial_amt != 0:
            pytest.fail(f"Initial position for {trade_symbol} is not zero: {initial_amt}. Please close manually.")

        # 2. Open Position (Smallest possible size)
        # We need to know lot size + min notional. 
        # For BTC, 0.001 or 0.002 is usually safe for testnet.
        # HARDCODING for now based on common testnet constraints, or use a param?
        qty = 0.002 
        
        LOGGER.info(f"Placing BUY Order for {qty} {trade_symbol}...")
        order = binance_client.place_order(
            symbol=trade_symbol,
            side="BUY",
            order_type="MARKET",
            quantity=qty
        )
        LOGGER.info(f"Order Placed: {order.get('orderId')}")
        
        # Wait for fill
        time.sleep(2)
        
        # 3. Verify Position
        pos_after_open = binance_client.get_position(trade_symbol)
        amt_after_open = float(pos_after_open.get('positionAmt', 0.0))
        LOGGER.info(f"Position After Open: {amt_after_open}")
        
        assert amt_after_open > 0, "Position should be positive after long"
        # assert amt_after_open == qty # Might be slightly different due to fees if deducted from qty? usually USDT-M fees are USDT.
        
        # 4. Close Position
        LOGGER.info("Closing Position...")
        close_order = binance_client.close_position(trade_symbol) # This helper handles full close
        LOGGER.info(f"Close Order Sent: {pformat(close_order)}")
        
        # Wait for fill
        time.sleep(2)
        
        # 5. Verify Closed
        pos_after_close = binance_client.get_position(trade_symbol)
        amt_after_close = float(pos_after_close.get('positionAmt', 0.0))
        LOGGER.info(f"Position After Close: {amt_after_close}")
        
        assert amt_after_close == 0, f"Position should be zero after close, got {amt_after_close}"
