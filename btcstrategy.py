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
        Generate trading signals based on calculated indicators.
        
        This method analyzes the calculated indicators and generates buy/sell signals
        based on a combination of EMA, RSI, ATR, and VWAP indicators.
        
        Returns:
            pd.DataFrame: DataFrame with calculated indicators and generated signals.
                          Signal values: 1.0 (buy), -1.0 (sell), 0.0 (hold)
        """
        try:
            # Get data with calculated indicators
            data = self.calculate_indicators()
            
            if data is None or data.empty:
                logger.warning("No data available to generate trading signals")
                return None
            
            # Initialize signal column with 0.0 (no signal/hold)
            data['signal'] = 0.0
            
            # Trading rules based on indicators
            for i in range(1, len(data)):
                # Rule 1: Price crosses above VWAP and RSI is above 50 -> Buy signal
                if (data['close'].iloc[i] > data['vwap'].iloc[i] and 
                    data['close'].iloc[i-1] <= data['vwap'].iloc[i-1] and 
                    data['rsi'].iloc[i] > 50):
                    data.loc[data.index[i], 'signal'] = 1.0
                
                # Rule 2: Price crosses below VWAP and RSI is below 50 -> Sell signal
                elif (data['close'].iloc[i] < data['vwap'].iloc[i] and 
                      data['close'].iloc[i-1] >= data['vwap'].iloc[i-1] and 
                      data['rsi'].iloc[i] < 50):
                    data.loc[data.index[i], 'signal'] = -1.0
                
                # Rule 3: Price is above EMA, but ATR is increasing significantly -> Sell signal (volatility exit)
                elif (data['close'].iloc[i] > data['ema'].iloc[i] and 
                      data['atr'].iloc[i] > data['atr'].iloc[i-1] * 1.5):
                    data.loc[data.index[i], 'signal'] = -1.0
                
                # Rule 4: Price is below EMA, but volatility is decreasing -> Buy signal (potential reversal)
                elif (data['close'].iloc[i] < data['ema'].iloc[i] and 
                      data['atr'].iloc[i] < data['atr'].iloc[i-1] * 0.7 and
                      data['rsi'].iloc[i] > 30):
                    data.loc[data.index[i], 'signal'] = 1.0
            
            # Add a column for position to track the current position based on signals
            data['position'] = 0
            
            # Initialize the first position based on the first signal
            if data['signal'].iloc[0] != 0:
                data.loc[data.index[0], 'position'] = data['signal'].iloc[0]
            
            # Update positions based on signals
            for i in range(1, len(data)):
                # If there's a new signal, update the position
                if data['signal'].iloc[i] != 0:
                    data.loc[data.index[i], 'position'] = data['signal'].iloc[i]
                # Otherwise, maintain the previous position
                else:
                    data.loc[data.index[i], 'position'] = data['position'].iloc[i-1]
            
            logger.info(f"Generated trading signals for {self.symbol} on {self.interval} timeframe")
            return data
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {e}", exc_info=True)
            return None


if __name__ == "__main__":
    import os
    client = Client(os.getenv("YOUR_API_KEY"), os.getenv("YOUR_API_SECRET"))
    btc_strategy = BTCStrategy(client, "BTCUSDT", Client.KLINE_INTERVAL_15MINUTE)
    
    try:
        # Generate trading signals
        signals_data = btc_strategy.run()
        
        if signals_data is not None:
            # Display the last 10 entries of the signals data
            print("\nLast 10 entries of trading signals:")
            print(signals_data[['timestamp', 'close', 'ema', 'rsi', 'vwap', 'atr', 'signal', 'position']].tail(10))
            
            # Count the number of buy and sell signals
            buy_signals = (signals_data['signal'] == 1.0).sum()
            sell_signals = (signals_data['signal'] == -1.0).sum()
            
            print(f"\nSignal summary:")
            print(f"Total data points: {len(signals_data)}")
            print(f"Buy signals: {buy_signals}")
            print(f"Sell signals: {sell_signals}")
            
    except Exception as e:
        print(f"An error occurred: {e}")
