import asyncio
import functools
import logging
from typing import Any, Awaitable, Callable, TypeVar

from exceptions import BaseTradingException

logger = logging.getLogger(__name__)


F = TypeVar("F", bound=Callable[..., Any])


def handle_error(func: F) -> F:
    """Decorator to log errors and raise ``BaseTradingException`` consistently."""

    @functools.wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await func(*args, **kwargs)  # type: ignore[misc]
        except BaseTradingException:
            raise
        except Exception as exc:
            logger.error("Error in %s: %s", func.__name__, exc, exc_info=True)
            raise BaseTradingException(f"Error in {func.__name__}: {exc}") from exc

    @functools.wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except BaseTradingException:
            raise
        except Exception as exc:
            logger.error("Error in %s: %s", func.__name__, exc, exc_info=True)
            raise BaseTradingException(f"Error in {func.__name__}: {exc}") from exc

    if asyncio.iscoroutinefunction(func):
        return async_wrapper  # type: ignore[return-value]
    return sync_wrapper  # type: ignore[return-value]

