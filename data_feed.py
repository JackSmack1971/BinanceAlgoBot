import logging
from binance.client import Client
import pandas as pd
from config import get_config

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
        self.symbol = symbol if symbol else get_config('default_symbol', "BTCUSDT")
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
        
        # Store data in the database
        from database.market_data_repository import MarketDataRepository
        market_data_repo = MarketDataRepository()
        for index, row in data.iterrows():
            market_data_repo.insert_market_data(
                symbol=self.symbol,
                interval=self.interval,
                timestamp=row['timestamp'],
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume'],
                close_time=row['close_time'],
                quote_asset_volume=row['quote_asset_volume'],
                trades=row['trades'],
                taker_buy_base=row['taker_buy_base'],
                taker_buy_quote=row['taker_buy_quote'],
                ignored=row['ignored']
            )
        
        return data
