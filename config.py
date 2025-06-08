"""
Configuration module for the Binance Algorithmic Trading Bot.

This module centralizes all configuration parameters used throughout the application,
providing a single source of truth for settings and making it easier to modify
parameters without changing code in multiple files.
"""

from configuration_service import TypedConfigurationService

# Binance related constants used across the project
BINANCE_CONSTANTS = {
    "KLINE_INTERVAL_1MINUTE": "1m",
    "KLINE_INTERVAL_3MINUTE": "3m",
    "KLINE_INTERVAL_5MINUTE": "5m",
    "KLINE_INTERVAL_15MINUTE": "15m",
    "KLINE_INTERVAL_30MINUTE": "30m",
    "KLINE_INTERVAL_1HOUR": "1h",
}

# Supported strategy types with short descriptions
STRATEGY_TYPES = {
    "btc": "BTC strategy",
    "ema_cross": "EMA crossover strategy",
    "macd": "MACD strategy",
}

config_service = TypedConfigurationService('config.json')

# Convenience constant for database URL
DATABASE_URL = config_service.get_config('database_url')

config_service.declare_config('database_url', str, "postgresql://user:password@host:port/database")
config_service.declare_config('use_testnet', bool, True)
config_service.declare_config('default_symbol', str, "BTCUSDT")
config_service.declare_config('default_interval', str, "15m")
config_service.declare_config('default_quantity', float, 0.001)
config_service.declare_config('ema_window', int, 14)
config_service.declare_config('rsi_window', int, 14)
config_service.declare_config('rsi_overbought', int, 70)
config_service.declare_config('rsi_oversold', int, 30)
config_service.declare_config('atr_window', int, 14)
config_service.declare_config('atr_volatility_factor', float, 1.5)
config_service.declare_config('vwap_window', int, 14)
config_service.declare_config('rsi_buy_threshold', int, 50)
config_service.declare_config('rsi_sell_threshold', int, 50)
config_service.declare_config('volatility_increase_factor', float, 1.5)
config_service.declare_config('volatility_decrease_factor', float, 0.7)
config_service.declare_config('max_risk_per_trade', float, 0.01)
config_service.declare_config('max_open_trades', int, 3)
config_service.declare_config('stop_loss_percentage', float, 0.02)
config_service.declare_config('take_profit_percentage', float, 0.04)
config_service.declare_config('logging_level', str, "INFO")
config_service.declare_config('logging_format', str, "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
config_service.declare_config('log_to_file', bool, True)
config_service.declare_config('log_file', str, "trading_bot.log")

def get_config(key: str):
    return config_service.get_config(key)

# Example usage:
# api_key = get_config('api_key')
# print(f"API Key: {api_key}")

# You can access configuration values like this:
# database_url = get_config('database_url', "postgresql://user:password@host:port/database")
# use_testnet = get_config('use_testnet', True)
# default_symbol = get_config('default_symbol', "BTCUSDT")
# default_interval = get_config('default_interval', "15m")
# default_quantity = get_config('default_quantity', 0.001)
# ema_window = get_config('ema_window', 14)
# rsi_window = get_config('rsi_window', 14)
# rsi_overbought = get_config('rsi_overbought', 70)
# rsi_oversold = get_config('rsi_oversold', 30)
# atr_window = get_config('atr_window', 14)
# atr_volatility_factor = get_config('atr_volatility_factor', 1.5)
# vwap_window = get_config('vwap_window', 14)
# rsi_buy_threshold = get_config('rsi_buy_threshold', 50)
# rsi_sell_threshold = get_config('rsi_sell_threshold', 50)
# volatility_increase_factor = get_config('volatility_increase_factor', 1.5)
# volatility_decrease_factor = get_config('volatility_decrease_factor', 0.7)
# max_risk_per_trade = get_config('max_risk_per_trade', 0.01)
# max_open_trades = get_config('max_open_trades', 3)
# stop_loss_percentage = get_config('stop_loss_percentage', 0.02)
# take_profit_percentage = get_config('take_profit_percentage', 0.04)
# logging_level = get_config('level', "INFO")
# logging_format = get_config('format', "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# log_to_file = get_config('log_to_file', True)
# log_file = get_config('log_file', "trading_bot.log")
