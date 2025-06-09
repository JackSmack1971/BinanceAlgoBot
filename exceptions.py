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

class DataRetrievalError(BaseTradingException):
    """Exception raised when there is an error retrieving data."""
    pass

class TradeExecutionError(BaseTradingException):
    """Exception raised when trade execution fails."""
    pass

class IndicatorServiceException(BaseTradingException):
    """Exception raised from indicator services."""
    pass

class MarketDataServiceException(BaseTradingException):
    """Exception raised from market data services."""
    pass

class PerformanceMetricsServiceException(BaseTradingException):
    """Exception raised from performance metrics services."""
    pass


class CredentialError(BaseTradingException):
    """Exception raised for credential handling failures."""
    pass
