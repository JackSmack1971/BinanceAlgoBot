import logging
import pandas as pd
from binance.client import Client
from data_feed import DataFeed
from indicators import TechnicalIndicators
from config import INDICATOR_CONFIG, STRATEGY_CONFIG, BINANCE_CONSTANTS

logger = logging.getLogger(__name__)

class BTCStrategy:
    def __init__(self, client: Client, symbol: str = None, interval: str = None,
                 ema_window: int = None, rsi_window: int = None, 
                 atr_window: int = None, vwap_window: int = None):
        """
        Initialize the BTC trading strategy.
        
        Args:
            client (Client): Binance API client
            symbol (str, optional): Trading symbol (e.g., 'BTCUSDT'). 
                                   Defaults to config value if None.
            interval (str, optional): Timeframe interval (e.g., '15m'). 
                                     Defaults to config value if None.
            ema_window (int, optional): EMA window. Defaults to config value if None.
            rsi_window (int, optional): RSI window. Defaults to config value if None.
            atr_window (int, optional): ATR window. Defaults to config value if None.
            vwap_window (int, optional): VWAP window. Defaults to config value if None.
        """
        from config import TRADING_CONFIG
        
        self.client = client
        self.symbol = symbol if symbol else TRADING_CONFIG["default_symbol"]
        self.interval = interval if interval else BINANCE_CONSTANTS["KLINE_INTERVAL_15MINUTE"]
        
        # Use provided parameters or defaults from config
        self.ema_window = ema_window if ema_window else INDICATOR_CONFIG["ema_window"]
        self.rsi_window = rsi_window if rsi_window else INDICATOR_CONFIG["rsi_window"]
        self.atr_window = atr_window if atr_window else INDICATOR_CONFIG["atr_window"]
        self.vwap_window = vwap_window if vwap_window else INDICATOR_CONFIG["vwap_window"]
        
        # Initialize dependencies
        self.data_feed = DataFeed(client, self.symbol, self.interval)
        self.indicators = TechnicalIndicators(
            ema_window=self.ema_window,
            rsi_window=self.rsi_window,
            atr_window=self.atr_window,
            vwap_window=self.vwap_window
        )
        
        logger.info(f"Initialized BTCStrategy with symbol={self.symbol}, interval={self.interval}")
    
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
            
            # Get configuration values
            rsi_buy = STRATEGY_CONFIG["rsi_buy_threshold"]
            rsi_sell = STRATEGY_CONFIG["rsi_sell_threshold"]
            volatility_increase = STRATEGY_CONFIG["volatility_increase_factor"]
            volatility_decrease = STRATEGY_CONFIG["volatility_decrease_factor"]
            rsi_oversold = INDICATOR_CONFIG["rsi_oversold"]
            
            # Trading rules based on indicators
            for i in range(1, len(data)):
                # Rule 1: Price crosses above VWAP and RSI is above threshold -> Buy signal
                if (data['close'].iloc[i] > data['vwap'].iloc[i] and 
                    data['close'].iloc[i-1] <= data['vwap'].iloc[i-1] and 
                    data['rsi'].iloc[i] > rsi_buy):
                    data.loc[data.index[i], 'signal'] = 1.0
                
                # Rule 2: Price crosses below VWAP and RSI is below threshold -> Sell signal
                elif (data['close'].iloc[i] < data['vwap'].iloc[i] and 
                      data['close'].iloc[i-1] >= data['vwap'].iloc[i-1] and 
                      data['rsi'].iloc[i] < rsi_sell):
                    data.loc[data.index[i], 'signal'] = -1.0
                
                # Rule 3: Price is above EMA, but ATR is increasing significantly -> Sell signal (volatility exit)
                elif (data['close'].iloc[i] > data['ema'].iloc[i] and 
                      data['atr'].iloc[i] > data['atr'].iloc[i-1] * volatility_increase):
                    data.loc[data.index[i], 'signal'] = -1.0
                
                # Rule 4: Price is below EMA, but volatility is decreasing -> Buy signal (potential reversal)
                elif (data['close'].iloc[i] < data['ema'].iloc[i] and 
                      data['atr'].iloc[i] < data['atr'].iloc[i-1] * volatility_decrease and
                      data['rsi'].iloc[i] > rsi_oversold):
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
    from config import LOGGING_CONFIG
    import logging.config
    
    # Configure logging
    logging.config.dictConfig({
        'version': 1,
        'formatters': {
            'standard': {
                'format': LOGGING_CONFIG['format']
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': LOGGING_CONFIG['level'],
                'formatter': 'standard',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.FileHandler',
                'level': LOGGING_CONFIG['level'],
                'formatter': 'standard',
                'filename': LOGGING_CONFIG['log_file'],
                'mode': 'a',
            }
        },
        'loggers': {
            '': {  # root logger
                'handlers': ['console', 'file'] if LOGGING_CONFIG['log_to_file'] else ['console'],
                'level': LOGGING_CONFIG['level'],
                'propagate': True
            }
        }
    })
    
    # Initialize the client and strategy
    client = Client(os.getenv("YOUR_API_KEY"), os.getenv("YOUR_API_SECRET"))
    btc_strategy = BTCStrategy(client)
    
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
        logger.error(f"An error occurred in main: {e}", exc_info=True)
