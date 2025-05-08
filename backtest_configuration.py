from binance.client import Client
from strategy_factory import Strategy
from data_source_strategy import DataSourceStrategy

class BacktestConfiguration:
    """
    Configuration class for backtesting.
    """

    def __init__(self, client: Client, strategy: Strategy, start_date: str, end_date: str, data_source: DataSourceStrategy, initial_capital: float = 10000.0, commission: float = 0.001):
        """
        Initializes the BacktestConfiguration.

        Args:
            client (Client): Binance API client.
            strategy (Strategy): Trading strategy to backtest.
            start_date (str): Start date for backtesting (format: 'YYYY-MM-DD').
            end_date (str): End date for backtesting (format: 'YYYY-MM-DD').
            data_source (DataSourceStrategy): Data source strategy.
            initial_capital (float, optional): Initial capital for the backtest. Defaults to 10000.0.
            commission (float, optional): Commission per trade. Defaults to 0.001.
        """
        self.client = client
        self.strategy = strategy
        self.start_date = start_date
        self.end_date = end_date
        self.data_source = data_source
        self.initial_capital = initial_capital
        self.commission = commission