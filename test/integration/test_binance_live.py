
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
        # Verify symbol validity via CCXT
        try:
            binance_client.exchange.load_markets()
            if symbol not in binance_client.exchange.markets:
                # CCXT might expect BTC/USDT:USDT
                alt_symbol = symbol + ":USDT" if ":USDT" not in symbol else symbol
                if alt_symbol in binance_client.exchange.markets:
                    symbol = alt_symbol
                else:
                    LOGGER.warning(f"Symbol {symbol} might be invalid for CCXT binanceusdm.")
        except Exception as e:
            LOGGER.warning(f"Failed to load markets: {e}")

        trade_symbol = symbol
        LOGGER.info(f"Starting Live Test on {trade_symbol}")
        
        try:
            # 1. Pre-check: Ensure empty position
            positions = binance_client.get_position(trade_symbol)
            
            initial_amt = 0.0
            if positions is None:
                LOGGER.warning(f"get_position({trade_symbol}) returned None (No position info found). Assuming 0.")
            else:
                initial_amt = float(positions.get('positionAmt', 0.0))
                
            if initial_amt != 0:
                pytest.fail(f"Initial position for {trade_symbol} is not zero: {initial_amt}. Please close manually.")

            # 2. Open Position (Smallest possible size)
            # We need to know lot size + min notional. 
            # For BTC, 0.001 or 0.002 is usually safe for testnet.
            qty = 0.002 
            
            LOGGER.info(f"Placing BUY Order for {qty} {trade_symbol}...")
            order = binance_client.submit_order(
                symbol=trade_symbol,
                side="BUY",
                order_type="MARKET",
                qty=qty
            )
            LOGGER.info(f"Order Response: {pformat(order)}")
            
            # Verify Order Status
            # CCXT status: 'open', 'closed', 'canceled', 'expired', 'rejected'
            status = order.get('status')
            if status not in ('FILLED', 'NEW', 'PARTIALLY_FILLED', 'closed', 'open'):
                 pytest.fail(f"Order failed with status {status}. Info: {order}")

            # Wait for position update (Retry loop)
            max_retries = 20
            amt_after_open = 0.0
            for i in range(max_retries):
                time.sleep(1)
                
                # Method A: Standard Wrapper
                positions = binance_client.get_position(trade_symbol)
                if positions:
                    amt_after_open = float(positions.get('positionAmt', 0.0))
                    if abs(amt_after_open) > 0:
                        break
                
                # Method B: Direct CCXT Symbol Fetch (Fallback)
                if abs(amt_after_open) == 0:
                    try:
                        # Normalize symbol for CCXT if formatting differs
                        # e.g. BTC/USDT:USDT -> BTCUSDT
                        # We try both raw and normalized
                        candidates = [trade_symbol]
                        if "/" in trade_symbol:
                             candidates.append(trade_symbol.replace("/", "").replace(":USDT", ""))
                        
                        raw_pos_list = binance_client.exchange.fetch_positions(candidates)
                        for p in raw_pos_list:
                            if float(p.get('contracts', 0) or p.get('info', {}).get('positionAmt', 0)) != 0:
                                amt_after_open = float(p.get('contracts', 0) or p.get('info', {}).get('positionAmt', 0))
                                LOGGER.info(f"Method B found position: {amt_after_open} in {p['symbol']}")
                                break
                    except Exception:
                        pass
                
                if abs(amt_after_open) > 0:
                    break

                LOGGER.warning(f"Waiting for position update... ({i+1}/{max_retries})")
            
            # 3. Verify Position
            if abs(amt_after_open) == 0:
                 LOGGER.warning(f"Position missing after OPEN for {trade_symbol}. Debugging...")
                 all_pos = binance_client.exchange.fetch_positions()
                 avail_syms = [p.get('symbol') for p in all_pos]
                 LOGGER.warning(f"Available symbols: {avail_syms}")
                 
                 # Also check open orders to see if it's stuck
                 open_orders = binance_client.list_orders(symbol=trade_symbol)
                 LOGGER.warning(f"Open Orders: {pformat(open_orders)}")
                 
                 balance = binance_client.account_overview()
                 LOGGER.warning(f"Balance Info: {pformat(balance)}")
                 
                 pytest.fail(f"Position not found after OPEN order. Order Status: {status}")
                 
            LOGGER.info(f"Position After Open: {amt_after_open}")
            
            assert amt_after_open > 0, "Position should be positive after long"
            
            # 4. Close Position
            LOGGER.info("Closing Position...")
            close_order = binance_client.close_position(trade_symbol) # This helper handles full close
            LOGGER.info(f"Close Order Sent: {pformat(close_order)}")
            
            # Wait for fill
            time.sleep(2)
            
            # 5. Verify Closed
            pos_after_close = binance_client.get_position(trade_symbol)
            amt_after_close = 0.0
            if pos_after_close:
                amt_after_close = float(pos_after_close.get('positionAmt', 0.0))
            LOGGER.info(f"Position After Close: {amt_after_close}")
            
            assert amt_after_close == 0, f"Position should be zero after close, got {amt_after_close}"

        finally:
            # Safety Cleanup: Cancel all open orders for this symbol
            try:
                LOGGER.info("Safety Cleanup: Cancelling open orders...")
                binance_client.exchange.cancel_all_orders(trade_symbol)
            except Exception as e:
                LOGGER.warning(f"Cleanup failed (might be fine if no orders): {e}")
