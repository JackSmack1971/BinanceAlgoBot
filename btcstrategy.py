import logging
import pandas as pd
from binance.client import Client
from data_feed import DataFeed
from indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

class BTCStrategy:
    def __init__(self, client: Client, symbol: str, interval: str, 
                 ema_window=14, rsi_window=14, atr_window=14, vwap_window=14):
        """
        Initialize the BTC trading strategy.
        
        Args:
            client (Client): Binance API client
            symbol (str): Trading symbol (e.g., 'BTCUSDT')
            interval (str): Timeframe interval (e.g., '15m')
            ema_window (int): Window period for EMA
            rsi_window (int): Window period for RSI
            atr_window (int): Window period for ATR
            vwap_window (int): Window period for VWAP
        """
        self.client = client
        self.symbol = symbol
        self.interval = interval
        self.data_feed = DataFeed(client, symbol, interval)
        self.indicators = TechnicalIndicators(
            ema_window=ema_window,
            rsi_window=rsi_window,
            atr_window=atr_window,
            vwap_window=vwap_window
        )
    
    def calculate_indicators(self):
        """
        Retrieve data and calculate all indicators.
        
        Returns:
            pd.DataFrame: DataFrame with all indicators calculated
        """
        try:
            data = self.data_feed.get_data()
            if data is not None:
                data = self.indicators.calculate_all(data)
            return data
        except Exception as e:
            logger.error(f"An error occurred while calculating indicators: {e}", exc_info=True)
            return None
    
    def run(self):
        """
        Generate trading signals based on indicators.
        
        Returns:
            pd.DataFrame: DataFrame with trading signals
        """
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
