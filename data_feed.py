import logging
from binance.client import Client
import pandas as pd
from config import TRADING_CONFIG

logger = logging.getLogger(__name__)

class DataRetrievalError(Exception):
    """Exception raised when there is an error retrieving data."""
    pass

class DataFeed:
    def __init__(self, client: Client, symbol: str = None, interval: str = None):
        """
        Initialize the data feed.
        
        Args:
            client (Client): Binance API client
            symbol (str, optional): Trading symbol (e.g., 'BTCUSDT'). 
                                   Defaults to config value if None.
            interval (str, optional): Timeframe interval (e.g., '15m'). 
                                     Defaults to config value if None.
        """
        from config import BINANCE_CONSTANTS
        
        self.client = client
        self.symbol = symbol if symbol else TRADING_CONFIG["default_symbol"]
        self.interval = interval if interval else BINANCE_CONSTANTS["KLINE_INTERVAL_15MINUTE"]
        
        logger.info(f"Initialized DataFeed with symbol={self.symbol}, interval={self.interval}")

    def get_data(self):
        """
        Get historical data from Binance API.
        
        Returns:
            pd.DataFrame: DataFrame containing historical market data
            
        Raises:
            DataRetrievalError: If there is an error retrieving data from Binance
        """
        try:
            logger.info(f"Retrieving data for {self.symbol} on {self.interval} timeframe")
            klines = self.client.get_klines(symbol=self.symbol, interval=self.interval)
        except Exception as e:
            logger.error(f"An error occurred: {e}", exc_info=True)
            raise DataRetrievalError(f"Could not retrieve data for symbol {self.symbol} and interval {self.interval}") from e

        data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignored'])
        data = data.astype(float)
        data.fillna(method='ffill', inplace=True)  # Handle missing data
        
        logger.debug(f"Retrieved {len(data)} data points")
        return data
