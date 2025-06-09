from abc import ABC, abstractmethod
from position_manager import PositionManager
from utils import handle_error

class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""

    @handle_error
    def __init__(self, client: object, symbol: str, interval: str, initial_balance: float, risk_per_trade: float):
        self.client = client
        self.symbol = symbol
        self.interval = interval
        self.position_manager = PositionManager(initial_balance, risk_per_trade)

    @abstractmethod
    @handle_error
    def calculate_indicators(self):
        """Calculate indicators required for the strategy."""
        pass

    @abstractmethod
    @handle_error
    def run(self):
        """Generate trading signals based on the strategy."""
        pass

    @abstractmethod
    @handle_error
    def open_position(self, side: str, price: float, size: float):
        """Open a position."""
        pass

    @abstractmethod
    @handle_error
    def close_position(self, price: float):
        """Close the current position."""
        pass
