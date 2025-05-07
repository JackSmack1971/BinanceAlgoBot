import logging
import functools

logger = logging.getLogger(__name__)

def handle_error(func):
    """
    Decorator to handle exceptions in strategy methods.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"An error occurred in {func.__name__}: {e}", exc_info=True)
            return None
    return wrapper