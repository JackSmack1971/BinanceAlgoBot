import logging
import functools
from exceptions import BaseTradingException

logger = logging.getLogger(__name__)


def handle_error(func):
    """Decorator to log errors and raise a BaseTradingException."""
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            log = logging.getLogger(func.__module__)
            log.error(f"Error in {func.__name__}: {e}", exc_info=True)
            raise BaseTradingException(f"Error in {func.__name__}: {e}") from e

    return wrapper

