"""
Configuration module for the Binance Algorithmic Trading Bot.

This module centralizes all configuration parameters used throughout the application,
providing a single source of truth for settings and making it easier to modify
parameters without changing code in multiple files.
"""

from configuration_service import ConfigurationService

config_service = ConfigurationService('config.json')

def get_config(key: str, default=None):
    return config_service.get_config(key, default)

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
