import logging
from binance.client import Client
import pandas as pd 

logger = logging.getLogger(__name__) 

class DataRetrievalError(Exception):
    """Exception raised when there is an error retrieving data."""
    pass 

class DataFeed:
    def __init__(self, client: Client, symbol: str, interval: str):
        self.client = client
        self.symbol = symbol
        self.interval = interval 

    def get_data(self):
        """Get historical data from Binance API"""
        try:
            klines = self.client.get_klines(symbol=self.symbol, interval=self.interval)
        except Exception as e:
            logger.error(f"An error occurred: {e}", exc_info=True)
            raise DataRetrievalError(f"Could not retrieve data for symbol {self.symbol} and interval {self.interval}") from e 

        data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignored'])
        data = data.astype(float)
        data.fillna(method='ffill', inplace=True)  # Handle missing data
        return data 

if __name__ == "__main__":
    import os
    client = Client(os.getenv("YOUR_API_KEY"), os.getenv("YOUR_API_SECRET"))
    data_feed = DataFeed(client, "BTCUSDT", Client.KLINE_INTERVAL_15MINUTE)
    try:
        print(data_feed.get_data())
    except DataRetrievalError as e:
        print(f"An error occurred while retrieving data: {e}")
