import logging
import pandas as pd
from binance.client import Client
from data_feed import DataFeed
from indicators import TechnicalIndicators
from config import get_config, BINANCE_CONSTANTS

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    import os
    from config import get_config
    import logging.config

    # Configure logging
    logging.config.dictConfig({
        'version': 1,
        'formatters': {
            'standard': {
                'format': get_config('logging_format')
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': get_config('logging_level'),
                'formatter': 'standard',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.FileHandler',
                'level': get_config('logging_level'),
                'formatter': 'standard',
                'filename': get_config('log_file'),
                'mode': 'a',
            }
        },
        'loggers': {
            '': {  # root logger
            'handlers': ['console', 'file'] if get_config('log_to_file') else ['console'],
                'level': get_config('logging_level'),
                'propagate': True
            }
        }
    })

    # Initialize the client and strategy
    client = Client(os.getenv("YOUR_API_KEY"), os.getenv("YOUR_API_SECRET"))
    # btc_strategy = BTCStrategy(client) # Removed BTCStrategy

    try:
        # Generate trading signals
        # signals_data = btc_strategy.run() # Removed BTCStrategy

        # if signals_data is not None: # Removed BTCStrategy
        #     # Display the last 10 entries of the signals data # Removed BTCStrategy
        #     print("\nLast 10 entries of trading signals:") # Removed BTCStrategy
        #     print(signals_data[['timestamp', 'close', 'ema', 'rsi', 'vwap', 'atr', 'signal', 'position']].tail(10)) # Removed BTCStrategy

        #     # Count the number of buy and sell signals # Removed BTCStrategy
        #     buy_signals = (signals_data['signal'] == 1.0).sum() # Removed BTCStrategy
        #     sell_signals = (signals_data['signal'] == -1.0).sum() # Removed BTCStrategy

        #     # print(f"\nSignal summary:") # Removed BTCStrategy
        #     # print(f"Total data points: {len(signals_data)}") # Removed BTCStrategy
        #     # print(f"Buy signals: {buy_signals}") # Removed BTCStrategy
        #     # print(f"Sell signals: {sell_signals}") # Removed BTCStrategy

        print("BTCStrategy has been removed from this file. Please use StrategyFactory to create and run strategies.")

    except Exception as e:
        logger.error(f"An error occurred in main: {e}", exc_info=True)
        raise StrategyError(f"An error occurred in main: {e}") from e
