import logging
import pandas as pd
import numpy as np
from binance.client import Client
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice 

EMA_WINDOW = 14
RSI_WINDOW = 14
ATR_WINDOW = 14
VWAP_WINDOW = 14 

logger = logging.getLogger(__name__) 

class DataRetrievalError(Exception):
    """Exception raised when there is an error retrieving data."""
    pass 

class BTCStrategy:
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
        return data 

    def calculate_ema(self, data):
        """Calculate EMA"""
        ema_indicator = EMAIndicator(close=data['close'], window=EMA_WINDOW, fillna=True)
        data['ema'] = ema_indicator.ema_indicator() 
        return data 

    def calculate_rsi(self, data):
        """Calculate RSI"""
        rsi_indicator = RSIIndicator(close=data['close'], window=RSI_WINDOW, fillna=True)
        data['rsi'] = rsi_indicator.rsi() 
        return data 

    def calculate_atr(self, data):
        """Calculate ATR"""
        atr_indicator = AverageTrueRange(high=data['high'], low=data['low'], close=data['close'], window=ATR_WINDOW, fillna=True)
        data['atr'] = atr_indicator.average_true_range() 
        return data 

    def calculate_vwap(self, data):
        """Calculate VWAP"""
        vwap_indicator = VolumeWeightedAveragePrice(high=data['high'], low=data['low'], close=data['close'], volume=data['volume'], window=VWAP_WINDOW, fillna=True)
        data['vwap'] = vwap_indicator.volume_weighted_average_price() 
        return data 

    def calculate_indicators(self):
        """Calculate all indicators"""
        data = self.get_data()
        if data is not None:
            data = self.calculate_ema(data)
            data = self.calculate_rsi(data)
            data = self.calculate_atr(data)
            data = self.calculate_vwap(data)
        return data 

if __name__ == "__main__":
    import os
    client = Client(os.getenv("YOUR_API_KEY"), os.getenv("YOUR_API_SECRET"))
    strategy = BTCStrategy(client, "BTCUSDT", Client.KLINE_INTERVAL_15MINUTE)
    try:
        print(strategy.calculate_indicators())
    except DataRetrievalError as e:
        print(f"An error occurred while retrieving data: {e}")
