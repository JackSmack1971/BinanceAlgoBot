import logging
import pandas as pd
import numpy as np
from binance.client import Client
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice
from data_feed import DataFeed

EMA_WINDOW = 14
RSI_WINDOW = 14
ATR_WINDOW = 14
VWAP_WINDOW = 14

logger = logging.getLogger(__name__)

class BTCStrategy:
    def __init__(self, client: Client, symbol: str, interval: str):
        self.client = client
        self.symbol = symbol
        self.interval = interval
        self.data_feed = DataFeed(client, symbol, interval)

    def calculate_ema(self, data):
        """Calculate EMA"""
        try:
            ema_indicator = EMAIndicator(close=data['close'], window=EMA_WINDOW, fillna=True)
            data['ema'] = ema_indicator.ema_indicator()
        except Exception as e:
            logger.error(f"An error occurred while calculating EMA: {e}", exc_info=True)
        return data

    def calculate_rsi(self, data):
        """Calculate RSI"""
        try:
            rsi_indicator = RSIIndicator(close=data['close'], window=RSI_WINDOW, fillna=True)
            data['rsi'] = rsi_indicator.rsi()
        except Exception as e:
            logger.error(f"An error occurred while calculating RSI: {e}", exc_info=True)
        return data

    def calculate_atr(self, data):
        """Calculate ATR"""
        try:
            atr_indicator = AverageTrueRange(high=data['high'], low=data['low'], close=data['close'], window=ATR_WINDOW, fillna=True)
            data['atr'] = atr_indicator.average_true_range()
        except Exception as e:
            logger.error(f"An error occurred while calculating ATR: {e}", exc_info=True)
        return data

    def calculate_vwap(self, data):
        """Calculate VWAP"""
        try:
            vwap_indicator = VolumeWeightedAveragePrice(high=data['high'], low=data['low'], close=data['close'], volume=data['volume'], window=VWAP_WINDOW, fillna=True)
            data['vwap'] = vwap_indicator.volume_weighted_average_price()
        except Exception as e:
            logger.error(f"An error occurred while calculating VWAP: {e}", exc_info=True)
        return data

    def calculate_indicators(self):
        """Calculate all indicators"""
        try:
            data = self.data_feed.get_data()
            if data is not None:
                data = self.calculate_ema(data)
                data = self.calculate_rsi(data)
                data = self.calculate_atr(data)
                data = self.calculate_vwap(data)
            return data
        except Exception as e:
            logger.error(f"An error occurred while calculating indicators: {e}", exc_info=True)
            return None
            
    def run(self):
        """Generate trading signals based on indicators"""
        data = self.calculate_indicators()
        if data is None or data.empty:
            logger.warning("No data to generate signals")
            return None
            
        # Initialize signal column
        data['signal'] = 0.0
        
        # Generate signals based on indicator values
        # Example strategy: Buy when price crosses above EMA and RSI > 50, Sell when price crosses below EMA and RSI < 50
        for i in range(1, len(data)):
            if data['close'].iloc[i] > data['ema'].iloc[i] and data['rsi'].iloc[i] > 50:
                data.loc[data.index[i], 'signal'] = 1.0  # Buy signal
            elif data['close'].iloc[i] < data['ema'].iloc[i] and data['rsi'].iloc[i] < 50:
                data.loc[data.index[i], 'signal'] = -1.0  # Sell signal
                
        return data

if __name__ == "__main__":
    import os
    client = Client(os.getenv("YOUR_API_KEY"), os.getenv("YOUR_API_SECRET"))
    btc_strategy = BTCStrategy(client, "BTCUSDT", Client.KLINE_INTERVAL_15MINUTE)
    try:
        data = btc_strategy.calculate_indicators()
        print(data)
        
        # Test signal generation
        signals = btc_strategy.run()
        if signals is not None:
            print(signals[['timestamp', 'close', 'ema', 'rsi', 'signal']].tail())
    except Exception as e:
        print(f"An error occurred: {e}")
