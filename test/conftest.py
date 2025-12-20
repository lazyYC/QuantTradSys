
import os
import pytest
import logging

from config.env import load_env
from brokers.binance import BinanceCredentials, BinanceUSDMClient
from notifier.dispatcher import _configure_binance_symbol

LOGGER = logging.getLogger("conftest")

def pytest_addoption(parser):
    parser.addoption(
        "--run-live", action="store_true", default=False, help="run live trading tests"
    )
    parser.addoption(
        "--symbol", action="store", default="BTC/USDT", help="Symbol to test with (default: BTC/USDT)"
    )

def pytest_configure(config):
    config.addinivalue_line("markers", "live: mark test as requiring live exchange connection")

def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-live"):
        # --run-live given in cli: do not skip live tests
        return
    skip_live = pytest.mark.skip(reason="need --run-live option to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)

@pytest.fixture(scope="session")
def env_setup():
    """Load environment variables once."""
    load_env()

@pytest.fixture
def binance_client(env_setup, pytestconfig):
    """
    Returns a configured BinanceUSDMClient IF API keys are present.
    Run tests marked @pytest.mark.live to use this.
    """
    if not pytestconfig.getoption("--run-live"):
        pytest.skip("Skipping live client creation without --run-live")

    api_key = os.getenv("BINANCE_API_KEY") or os.getenv("BINANCE_API_KEY_ID")
    api_secret = os.getenv("BINANCE_API_SECRET") or os.getenv("BINANCE_PRIVATE_KEY")
    
    if not api_key or not api_secret:
        # Debug info
        env_loc = os.getenv("QTS_ENV_PATH", "default")
        pytest.fail(f"Credentials not found. Checked BINANCE_API_KEY/ID. Env path: {env_loc}")

    # Use sandbox by default for tests unless explicitly overridden in env?
    # User likely wants to test on whatever env is configured (Testnet or Real).
    # We should trust the env vars, but maybe log a warning.
    is_sandbox = os.getenv("BINANCE_SANDBOX", "true").lower() in ("true", "1", "yes")
    
    LOGGER.warning(f"Creating Binance Client (Sandbox={is_sandbox})")
    
    client = BinanceUSDMClient(
        BinanceCredentials(api_key, api_secret),
        sandbox=is_sandbox
    )
    _configure_binance_symbol(client)
    return client

@pytest.fixture
def test_symbol(pytestconfig):
    return pytestconfig.getoption("--symbol")
