import logging
import pandas as pd
from typing import Dict, Any
from backtest_configuration import BacktestConfiguration
from backtest_helper import BacktestHelper
from strategies.base_strategy import BaseStrategy
from binance.client import Client
from datetime import datetime
from exceptions import BaseTradingException

logger = logging.getLogger(__name__)

class BacktestExecutor:
    """
    Executes the backtest for a given strategy and historical data.
    """

    from utils import handle_error

    @handle_error
    def __init__(self, config: BacktestConfiguration):
        """
        Initializes the BacktestExecutor with a BacktestConfiguration object.

        Args:
            config (BacktestConfiguration): Backtest configuration object.
        """
        self.config = config
        logger.info(f"Initialized BacktestExecutor for {config.strategy.__class__.__name__} on {config.strategy.symbol}/{config.strategy.interval}")

    def run(self) -> pd.DataFrame:
        """
        Runs the backtest for the specified period.

        Returns:
            pd.DataFrame: DataFrame with backtesting results.
        """
        try:
            logger.info(f"Running backtest from {self.config.start_date} to {self.config.end_date}")

            # Fetch historical data using BacktestHelper
            data = BacktestHelper.fetch_historical_data(self.config)

            if data.empty:
                logger.error(f"No historical data available for {self.config.strategy.symbol} from {self.config.start_date} to {self.config.end_date}")
                return pd.DataFrame()

            # Apply the strategy using BacktestHelper
            signals = BacktestHelper.apply_strategy(data, self.config.strategy)

            if signals is None or signals.empty:
                logger.error(f"No signals generated for {self.config.strategy.symbol}")
                return pd.DataFrame()

            logger.info(f"Backtest completed for {self.config.strategy.symbol} from {self.config.start_date} to {self.config.end_date}")
            return signals

        except Exception as e:
            logger.error(f"Error during backtesting: {e}", exc_info=True)
            raise BaseTradingException(f"Error during backtesting: {e}") from e
            return pd.DataFrame()