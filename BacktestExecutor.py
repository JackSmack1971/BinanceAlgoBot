import logging
import pandas as pd
from typing import Dict, Any
from binance.client import Client
from strategy_factory import Strategy
from datetime import datetime

logger = logging.getLogger(__name__)

class BacktestExecutor:
    """
    Executes the backtest for a given strategy and historical data.
    """

    def __init__(self, client: Client, strategy: Strategy):
        """
        Initializes the BacktestExecutor with a Binance client and a trading strategy.

        Args:
            client (Client): Binance API client.
            strategy (Strategy): Trading strategy to backtest.
        """
        self.client = client
        self.strategy = strategy
        logger.info(f"Initialized BacktestExecutor for {strategy.__class__.__name__} on {strategy.symbol}/{strategy.interval}")

    def run(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Runs the backtest for the specified period.

        Args:
            start_date (str): Start date for backtesting (format: 'YYYY-MM-DD').
            end_date (str): End date for backtesting (format: 'YYYY-MM-DD').

        Returns:
            pd.DataFrame: DataFrame with backtesting results.
        """
        try:
            logger.info(f"Running backtest from {start_date} to {end_date}")

            # Convert dates to milliseconds timestamp for Binance API
            start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
            end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)

            # Get historical data from Binance
            klines = self.client.get_historical_klines(
                symbol=self.strategy.symbol,
                interval=self.strategy.interval,
                start_str=start_ts,
                end_str=end_ts
            )

            if not klines:
                logger.error(f"No historical data available for {self.strategy.symbol} from {start_date} to {end_date}")
                return pd.DataFrame()

            # Convert to DataFrame
            data = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'trades',
                'taker_buy_base', 'taker_buy_quote', 'ignored'
            ])
            data = data.astype(float)

            # Convert timestamp to datetime
            data['datetime'] = pd.to_datetime(data['timestamp'], unit='ms')
            data.set_index('datetime', inplace=True)

            # Apply the strategy to generate signals
            # We'll override the strategy's data_feed to use our historical data
            self.strategy.data_feed.get_data = lambda: data

            # Calculate indicators and generate signals
            signals = self.strategy.run()

            if signals is None or signals.empty:
                logger.error(f"No signals generated for {self.strategy.symbol}")
                return pd.DataFrame()
            
            logger.info(f"Backtest completed for {self.strategy.symbol} from {start_date} to {end_date}")
            return signals

        except Exception as e:
            logger.error(f"Error during backtesting: {e}", exc_info=True)
            return pd.DataFrame()