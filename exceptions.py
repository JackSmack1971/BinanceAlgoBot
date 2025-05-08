class BaseTradingException(Exception):
    """Base class for all trading related exceptions."""
    pass

class ConfigurationError(BaseTradingException):
    """Exception raised for configuration errors."""
    pass

class DataError(BaseTradingException):
    """Exception raised for data related errors."""
    pass

class ExchangeError(BaseTradingException):
    """Exception raised for exchange related errors."""
    pass

class StrategyError(BaseTradingException):
    """Exception raised for strategy related errors."""
    pass

class RiskError(BaseTradingException):
    """Exception raised for risk management related errors."""
    pass

class OrderError(BaseTradingException):
    """Exception raised for order related errors."""
    pass