import logging
import pandas as pd
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice
from config import INDICATOR_CONFIG

logger = logging.getLogger(__name__)

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
        self.ema_window = ema_window if ema_window else INDICATOR_CONFIG["ema_window"]
        self.rsi_window = rsi_window if rsi_window else INDICATOR_CONFIG["rsi_window"]
        self.atr_window = atr_window if atr_window else INDICATOR_CONFIG["atr_window"]
        self.vwap_window = vwap_window if vwap_window else INDICATOR_CONFIG["vwap_window"]
    
    def calculate_ema(self, data):
        """
        Calculate Exponential Moving Average (EMA).
        
        Args:
            data (pd.DataFrame): DataFrame containing 'close' prices
            
        Returns:
            pd.DataFrame: DataFrame with added 'ema' column
        """
        try:
            ema_indicator = EMAIndicator(
                close=data['close'], 
                window=self.ema_window, 
                fillna=True
            )
            data['ema'] = ema_indicator.ema_indicator()
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
        return data
