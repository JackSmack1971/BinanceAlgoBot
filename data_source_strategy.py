import pandas as pd
from abc import ABC, abstractmethod
from typing import Protocol
import logging
from exceptions import DataError

class DataSourceStrategy(Protocol):
    """
    Interface for data source strategies.
    """

    @abstractmethod
    def get_data(self, symbol: str, interval: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetches data from the data source.

        Args:
            symbol (str): Trading symbol.
            interval (str): Trading interval.
            start_date (str): Start date for data.
            end_date (str): End date for data.

        Returns:
            pd.DataFrame: DataFrame with historical data.
        """
        pass


from utils import handle_error

logger = logging.getLogger(__name__)

class BinanceDataSourceStrategy:
    """
    Data source strategy for Binance.
    """

    @handle_error
    def __init__(self, client):
        self.client = client
    def get_data(self, symbol: str, interval: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetches historical data from Binance.

        Args:
            symbol (str): Trading symbol.
            interval (str): Trading interval.
            start_date (str): Start date for data.
            end_date (str): End date for data.

        Returns:
            pd.DataFrame: DataFrame with historical data.
        """
        try:
            # Convert dates to milliseconds timestamp for Binance API
            start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
            end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)

            # Get historical data from Binance
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_ts,
                end_str=end_ts
            )

            if not klines:
                print(f"No historical data available for {symbol} from {start_date} to {end_date}")
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

            return data

        except Exception as e:
            logger.error(f"Error fetching historical data: {e}", exc_info=True)
            raise DataError(f"Error fetching historical data: {e}") from e
            return pd.DataFrame()