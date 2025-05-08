from abc import ABC, abstractmethod
from position_manager import PositionManager
from utils import handle_error

class Strategy(ABC):
    """
    Abstract base class for all trading strategies.
    """

    @handle_error
    def __init__(self, client: object, symbol: str, interval: str, initial_balance: float, risk_per_trade: float):
        """
        Initialize the strategy.

        Args:
            client (Client): Binance API client
            symbol (str): Trading symbol
            interval (str): Timeframe interval
            initial_balance (float): Initial trading balance
            risk_per_trade (float): Risk per trade as a fraction of the balance
        """
        self.client = client
        self.symbol = symbol
        self.interval = interval
        self.position_manager = PositionManager(initial_balance, risk_per_trade)

    @abstractmethod
    @handle_error
    def calculate_indicators(self):
        """
        Calculate indicators required for the strategy.

        Returns:
            pd.DataFrame: DataFrame with calculated indicators
        """
        pass

    @abstractmethod
    @handle_error
    def run(self):
        """
        Generate trading signals based on the strategy.

        Returns:
            pd.DataFrame: DataFrame with trading signals
        """
        pass

    @abstractmethod
    @handle_error
    def open_position(self, side: str, price: float, size: float):
        """
        Open a position.

        Args:
            side (str): "buy" or "sell"
            price (float): Entry price
            size (float): Position size
        """
        pass

    @abstractmethod
    @handle_error
    def close_position(self, price: float):
        """
        Close the current position.

        Args:
            price (float): Exit price
        """
        pass