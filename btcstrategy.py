import pandas as pd
import numpy as np
from binance.client import Client
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice 

class BTCStrategy:
    def __init__(self, client: Client, symbol: str, interval: str):
        self.client = client
        self.symbol = symbol
        self.interval = interval 

    def get_data(self):
        """Get historical data from Binance API"""
        try:
            klines = self.client.get_klines(symbol=self.symbol, interval=self.interval)
       Apologies for the abrupt cut-off. Here is the continuation of the `btcstrategy.py` file:
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

        data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignored'])
        data = data.astype(float)
        return data 

    def calculate_ema(self, data):
        """Calculate EMA"""
        ema_indicator = EMAIndicator(close=data['close'], window=14, fillna=True)
        data['ema'] = ema_indicator.ema_indicator() 
        return data

    def calculate_rsi(self, data):
        """Calculate RSI"""
        rsi_indicator = RSIIndicator(close=data['close'], window=14, fillna=True)
        data['rsi'] = rsi_indicator.rsi() 
        return data

    def calculate_atr(self, data):
        """Calculate ATR"""
        atr_indicator = AverageTrueRange(high=data['high'], low=data['low'], close=data['close'], window=14, fillna=True)
        data['atr'] = atr_indicator.average_true_range() 
        return data

    def calculate_vwap(self, data):
        """Calculate VWAP"""
        vwap_indicator = VolumeWeightedAveragePrice(high=data['high'], low=data['low'], close=data['close'], volume=data['volume'], window=14, fillna=True)
        data['vwap'] = vwap_indicator.volume_weighted_average_price() 
        return data

    def calculate_indicators(self, data):
        """Calculate EMA, RSI, ATR, and VWAP indicators"""
        data = self.calculate_ema(data)
        data = self.calculate_rsi(data)
        data = self.calculate_atr(data)
        data = self.calculate_vwap(data)
        return data 

    def generate_signals(self, data):
        """Generate trading signals based on indicators"""
        # Create signals DataFrame
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0.0 

        # Create signal conditions
        long_condition = (data['ema'] > data['vwap']) & (data['rsi'] < 30)
        short_condition = (data['ema'] < data['vwap']) & (data['rsi'] > 70) 

        # Generate signals
        signals.loc[long_condition, 'signal'] = 1.0
        signals.loc[short_condition, 'signal'] = -1.0 

        return signals 

    def run(self):
        """Run the strategy"""
        data = self.get_data()
        if data is not None:
            data = self.calculate_indicators(data)
            signals = self.generate_signals(data)
            return signals 

if __name__ == "__main__":
    client = Client("YOUR_API_KEY", "YOUR_API_SECRET")
    strategy = BTCStrategy(client, "BTCUSDT", Client.KLINE_INTERVAL_15MINUTE)
    signals = strategy.run()
    print(signals)
