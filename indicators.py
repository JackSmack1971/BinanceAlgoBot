import logging
import pandas as pd
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Class for calculating various technical indicators for trading."""
    
    def __init__(self, ema_window=14, rsi_window=14, atr_window=14, vwap_window=14):
        """
        Initialize the indicators with configurable window parameters.
        
        Args:
            ema_window (int): Window period for EMA calculation
            rsi_window (int): Window period for RSI calculation
            atr_window (int): Window period for ATR calculation
            vwap_window (int): Window period for VWAP calculation
        """
        self.ema_window = ema_window
        self.rsi_window = rsi_window
        self.atr_window = atr_window
        self.vwap_window = vwap_window
    
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


if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Generate sample data
    dates = pd.date_range('2023-01-01', periods=100)
    data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.normal(100, 5, 100),
        'high': np.random.normal(102, 5, 100),
        'low': np.random.normal(98, 5, 100),
        'close': np.random.normal(101, 5, 100),
        'volume': np.random.normal(1000, 200, 100)
    })
    
    # Calculate indicators
    indicators = TechnicalIndicators()
    result = indicators.calculate_all(data)
    print(result.tail())
