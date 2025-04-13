"""
Configuration module for the Binance Algorithmic Trading Bot.

This module centralizes all configuration parameters used throughout the application,
providing a single source of truth for settings and making it easier to modify
parameters without changing code in multiple files.
"""

# API Configuration
API_CONFIG = {
    "use_testnet": True,  # Set to False for production
}

# Trading Pair Configuration
TRADING_CONFIG = {
    "default_symbol": "BTCUSDT",
    "default_interval": "15m",  # 15 minutes
    "default_quantity": 0.001,  # Default trade size in BTC
}

# Technical Indicator Parameters
INDICATOR_CONFIG = {
    # Exponential Moving Average
    "ema_window": 14,
    
    # Relative Strength Index
    "rsi_window": 14,
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    
    # Average True Range
    "atr_window": 14,
    "atr_volatility_factor": 1.5,  # Factor to determine significant volatility change
    
    # Volume Weighted Average Price
    "vwap_window": 14,
}

# Strategy Parameters
STRATEGY_CONFIG = {
    # Signal thresholds
    "rsi_buy_threshold": 50,
    "rsi_sell_threshold": 50,
    
    # Volatility thresholds
    "volatility_increase_factor": 1.5,  # ATR increase that triggers a volatility exit
    "volatility_decrease_factor": 0.7,  # ATR decrease that might signal a reversal
}

# Risk Management Parameters
RISK_CONFIG = {
    "max_risk_per_trade": 0.01,  # 1% of account balance per trade
    "max_open_trades": 3,
    "stop_loss_percentage": 0.02,  # 2% stop loss
    "take_profit_percentage": 0.04,  # 4% take profit
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_to_file": True,
    "log_file": "trading_bot.log",
}

# Binance API Timeframes (for reference)
TIMEFRAMES = {
    "1m": "1 minute",
    "3m": "3 minutes",
    "5m": "5 minutes",
    "15m": "15 minutes",
    "30m": "30 minutes",
    "1h": "1 hour",
    "2h": "2 hours",
    "4h": "4 hours",
    "6h": "6 hours",
    "8h": "8 hours",
    "12h": "12 hours",
    "1d": "1 day",
    "3d": "3 days",
    "1w": "1 week",
    "1M": "1 month",
}

# Define constants used by the Binance Client
# These match the constants defined in the Binance API Client
BINANCE_CONSTANTS = {
    "KLINE_INTERVAL_1MINUTE": "1m",
    "KLINE_INTERVAL_3MINUTE": "3m",
    "KLINE_INTERVAL_5MINUTE": "5m",
    "KLINE_INTERVAL_15MINUTE": "15m",
    "KLINE_INTERVAL_30MINUTE": "30m",
    "KLINE_INTERVAL_1HOUR": "1h",
    "KLINE_INTERVAL_2HOUR": "2h",
    "KLINE_INTERVAL_4HOUR": "4h",
    "KLINE_INTERVAL_6HOUR": "6h",
    "KLINE_INTERVAL_8HOUR": "8h",
    "KLINE_INTERVAL_12HOUR": "12h",
    "KLINE_INTERVAL_1DAY": "1d",
    "KLINE_INTERVAL_3DAY": "3d",
    "KLINE_INTERVAL_1WEEK": "1w",
    "KLINE_INTERVAL_1MONTH": "1M",
}
