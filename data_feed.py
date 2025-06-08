import logging
from binance.client import Client
import pandas as pd
from config import get_config
from exceptions import BaseTradingException, DataError, DataRetrievalError

logger = logging.getLogger(__name__)

from utils import handle_error

class DataFeed:
    @handle_error
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
        self.symbol = symbol if symbol else get_config('default_symbol')
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
            logger.info("Retrieving data for %s on %s timeframe", self.symbol, self.interval)
            klines = self.client.get_klines(symbol=self.symbol, interval=self.interval)
        except Exception as exc:
            logger.warning("Data retrieval failed: %s", exc)
            raise DataRetrievalError(
                f"Could not retrieve data for symbol {self.symbol} and interval {self.interval}"
            ) from exc

        data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignored'])
        data = data.astype(float)
        data.fillna(method='ffill', inplace=True)  # Handle missing data
        
        logger.debug(f"Retrieved {len(data)} data points")
        
        # Store data in the database
        from database.market_data_repository import MarketDataRepository
        market_data_repo = MarketDataRepository()

        market_data_list = []
        for _, row in data.iterrows():
            market_data = {
                'symbol': self.symbol,
                'interval': self.interval,
                'timestamp': row['timestamp'],
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume'],
                'close_time': row['close_time'],
                'quote_asset_volume': row['quote_asset_volume'],
                'trades': row['trades'],
                'taker_buy_base': row['taker_buy_base'],
                'taker_buy_quote': row['taker_buy_quote'],
                'ignored': row['ignored']
            }
            market_data_list.append(market_data)

        try:
            market_data_repo.insert_market_data(market_data_list)
        except DataError as exc:
            logger.error("Database error storing market data: %s", exc)
            raise
        
        return data
