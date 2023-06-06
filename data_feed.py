from binance.client import Client
import pandas as pd 

class DataFeed:
    def __init__(self, client: Client, symbol: str, interval: str):
        self.client = client
        self.symbol = symbol
        self.interval = interval 

    def get_data(self):
        """Get historical data from Binance API"""
        klines = self.client.get_klines(symbol=self.symbol, interval=self.interval)
        data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignored'])
        data = data.astype(float)
        return data 

if __name__ == "__main__":
    client = Client("YOUR_API_KEY", "YOUR_API_SECRET")
    data_feed = DataFeed(client, "BTCUSDT", Client.KLINE_INTERVAL_15MINUTE)
    data = data_feed.get_data()
    print(data)
