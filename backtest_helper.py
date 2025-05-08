import logging
import pandas as pd
from strategy_factory import Strategy
from typing import Dict, Any
from backtest_configuration import BacktestConfiguration

logger = logging.getLogger(__name__)

class BacktestHelper:
    """
    Helper class for backtesting functionality.
    """

    @staticmethod
    def fetch_historical_data(config: BacktestConfiguration) -> pd.DataFrame:
        """
        Fetches historical data from the data source.

        Args:
            config (BacktestConfiguration): Backtest configuration object.

        Returns:
            pd.DataFrame: DataFrame with historical data.
        """
        try:
            logger.info(f"Fetching historical data for {config.strategy.symbol} from {config.start_date} to {config.end_date}")

            # Get historical data from the data source
            data = config.data_source.get_data(config.strategy.symbol, config.strategy.interval, config.start_date, config.end_date)

            if data.empty:
                logger.error(f"No historical data available for {config.strategy.symbol} from {config.start_date} to {config.end_date}")
                return pd.DataFrame()

            return data

        except Exception as e:
            logger.error(f"Error fetching historical data: {e}", exc_info=True)
            raise DataError(f"Error fetching historical data: {e}") from e
            return pd.DataFrame()

    @staticmethod
    def apply_strategy(data: pd.DataFrame, strategy: Strategy) -> pd.DataFrame:
        """
        Applies the strategy to generate signals.

        Args:
            data (pd.DataFrame): DataFrame with historical data.
            strategy (Strategy): Trading strategy to apply.

        Returns:
            pd.DataFrame: DataFrame with trading signals.
        """
        try:
            logger.info(f"Applying strategy {strategy.__class__.__name__} to historical data")

            # We'll override the strategy's data_feed to use our historical data
            strategy.data_feed.get_data = lambda: data

            # Calculate indicators and generate signals
            signals = strategy.run()

            if signals is None or signals.empty:
                logger.error(f"No signals generated for {strategy.symbol}")
                return pd.DataFrame()

            return signals

        except Exception as e:
            logger.error(f"Error applying strategy: {e}", exc_info=True)
            raise StrategyError(f"Error applying strategy: {e}") from e
            return pd.DataFrame()