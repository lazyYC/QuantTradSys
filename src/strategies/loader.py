import importlib
import inspect
import logging
from typing import Type

from strategies.base import BaseStrategy

LOGGER = logging.getLogger(__name__)

def load_strategy_class(strategy_name: str) -> Type[BaseStrategy]:
    """
    Dynamically load a strategy class based on the strategy name.
    
    Args:
        strategy_name: The name of the strategy (e.g., "star_xgb").
                       This should correspond to a package in `strategies/`.
                       
    Returns:
        The strategy class (subclass of BaseStrategy).
        
    Raises:
        ValueError: If the strategy cannot be found or loaded.
    """
    try:
        # Try to import the adapter module
        # Convention: strategies.{strategy_name}.adapter
        module_path = f"strategies.{strategy_name}.adapter"
        module = importlib.import_module(module_path)
        
        # Find the class that inherits from BaseStrategy
        for name, obj in inspect.getmembers(module):
            if (
                inspect.isclass(obj)
                and issubclass(obj, BaseStrategy)
                and obj is not BaseStrategy
            ):
                LOGGER.info(f"Loaded strategy class: {name} from {module_path}")
                return obj
                
        raise ValueError(f"No BaseStrategy subclass found in {module_path}")
        
    except ImportError as e:
        raise ValueError(f"Could not load strategy '{strategy_name}': {e}")


def load_strategy_runtime(strategy_name: str):
    """
    Dynamically load the runtime module of a strategy.
    Convention: strategies.{strategy_name}.runtime
    
    The runtime module is expected to export:
    - generate_realtime_signal
    - load_star_model (or generic load_model)
    - RuntimeState class
    - IndicatorParams class
    - ModelParams class
    """
    try:
        module_path = f"strategies.{strategy_name}.runtime"
        module = importlib.import_module(module_path)
        return module
    except ImportError as e:
        raise ValueError(f"Could not load runtime for strategy '{strategy_name}': {e}")

