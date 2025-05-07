import logging
import pandas as pd
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice
from config import get_config

logger = logging.getLogger(__name__)

import time

class TechnicalIndicators:
    """Class for calculating various technical indicators for trading."""

    def __init__(self, ema_window=None, rsi_window=None, atr_window=None, vwap_window=None):
        """
        Initialize the indicators with configurable window parameters.
        
        Args:
            ema_window (int, optional): Window period for EMA calculation. 
                                       Defaults to config value if None.
            rsi_window (int, optional): Window period for RSI calculation. 
                                       Defaults to config value if None.
            atr_window (int, optional): Window period for ATR calculation. 
                                       Defaults to config value if None.
            vwap_window (int, optional): Window period for VWAP calculation. 
                                        Defaults to config value if None.
        """
        # Use provided parameters or defaults from config
        self.ema_window = ema_window if ema_window else get_config('ema_window', 14)
        self.rsi_window = rsi_window if rsi_window else get_config('rsi_window', 14)
        self.atr_window = atr_window if atr_window else get_config('atr_window', 14)
        self.vwap_window = vwap_window if vwap_window else get_config('vwap_window', 14)


class CachedIndicators(TechnicalIndicators):
    """
    Class for calculating technical indicators with caching.
    """

    def __init__(self, cache_expiry=60, *args, **kwargs):
        """
        Initialize the cached indicators with a cache expiry time.

        Args:
            cache_expiry (int, optional): Time in seconds before the cache expires. Defaults to 60.
        """
        super().__init__(*args, **kwargs)
        self.cache = {}
        self.cache_expiry = cache_expiry

    def calculate_ema(self, data):
        """
        Calculate Exponential Moving Average (EMA) with caching.

        Args:
            data (pd.DataFrame): DataFrame containing 'close' prices

        Returns:
            pd.DataFrame: DataFrame with added 'ema' column
        """
        cache_key = f"ema_{self.ema_window}_{len(data)}"
        if cache_key in self.cache:
            cached_value, cache_time = self.cache[cache_key]
            if time.time() - cache_time < self.cache_expiry:
                logger.info(f"Returning cached EMA for window {self.ema_window}")
                data['ema'] = cached_value
                return data
            else:
                logger.info(f"Cache expired for EMA with window {self.ema_window}, recalculating...")
        try:
            ema_indicator = EMAIndicator(
                close=data['close'],
                window=self.ema_window,
                fillna=True
            )
            ema = ema_indicator.ema_indicator()
            data['ema'] = ema
            self.cache[cache_key] = (ema, time.time())  # Cache the EMA values and the time
            return data
        except Exception as e:
            logger.error(f"An error occurred while calculating EMA: {e}", exc_info=True)
            return data
    
    def calculate_rsi(self, data):
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            data (pd.DataFrame): DataFrame containing 'close' prices
            
        Returns:
            pd.DataFrame: DataFrame with added 'rsi' column
        """
        try:
            rsi_indicator = RSIIndicator(
                close=data['close'], 
                window=self.rsi_window, 
                fillna=True
            )
            data['rsi'] = rsi_indicator.rsi()
            return data
        except Exception as e:
            logger.error(f"An error occurred while calculating RSI: {e}", exc_info=True)
            return data
    
    def calculate_atr(self, data):
        """
        Calculate Average True Range (ATR).
        
        Args:
            data (pd.DataFrame): DataFrame containing 'high', 'low', and 'close' prices
            
        Returns:
            pd.DataFrame: DataFrame with added 'atr' column
        """
        try:
            atr_indicator = AverageTrueRange(
                high=data['high'], 
                low=data['low'], 
                close=data['close'], 
                window=self.atr_window, 
                fillna=True
            )
            data['atr'] = atr_indicator.average_true_range()
            return data
        except Exception as e:
            logger.error(f"An error occurred while calculating ATR: {e}", exc_info=True)
            return data
    
    def calculate_vwap(self, data):
        """
        Calculate Volume Weighted Average Price (VWAP).
        
        Args:
            data (pd.DataFrame): DataFrame containing 'high', 'low', 'close', and 'volume' data
            
        Returns:
            pd.DataFrame: DataFrame with added 'vwap' column
        """
        try:
            vwap_indicator = VolumeWeightedAveragePrice(
                high=data['high'], 
                low=data['low'], 
                close=data['close'], 
                volume=data['volume'], 
                window=self.vwap_window, 
                fillna=True
            )
            data['vwap'] = vwap_indicator.volume_weighted_average_price()
            return data
        except Exception as e:
            logger.error(f"An error occurred while calculating VWAP: {e}", exc_info=True)
            return data
    
    def calculate_all(self, data):
        """
        Calculate all indicators at once.
        
        Args:
            data (pd.DataFrame): DataFrame containing required OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with all indicators added
        """
        if data is not None and not data.empty:
            data = self.calculate_ema(data)
            data = self.calculate_rsi(data)
            data = self.calculate_atr(data)
            data = self.calculate_vwap(data)

            # Store indicators in the database
            from database.indicator_repository import IndicatorRepository
            indicator_repo = IndicatorRepository()
            for index, row in data.iterrows():
                # Assuming 'timestamp' column exists in the DataFrame and corresponds to market_data
                # You might need to adjust the query based on your actual data structure
                from database.market_data_repository import MarketDataRepository
                market_data_repo = MarketDataRepository()
                market_data = market_data_repo.get_market_data(symbol="BTCUSDT", interval="15m", start_time=row['timestamp'], end_time=row['timestamp'])
                if market_data:
                    market_data_id = market_data[0][0]  # Assuming the first column is the ID
                    indicator_repo.insert_indicator(
                        market_data_id=market_data_id,
                        ema=row.get('ema'),
                        rsi=row.get('rsi'),
                        atr=row.get('atr'),
                        vwap=row.get('vwap')
                    )
        return data
