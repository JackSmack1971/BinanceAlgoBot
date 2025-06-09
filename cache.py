import os
import json
import logging
from functools import wraps
from typing import Any, Callable, Awaitable

import redis

logger = logging.getLogger(__name__)

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

F = Callable[..., Awaitable[Any]]

def cache_result(ttl: int = 300) -> Callable[[F], F]:
    """Cache async function results in Redis."""
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            try:
                cached = redis_client.get(cache_key)
                if cached:
                    return json.loads(cached)
                result = await func(*args, **kwargs)
                redis_client.setex(cache_key, ttl, json.dumps(result))
                return result
            except Exception as exc:  # pragma: no cover - redis errors
                logger.error("Cache error: %s", exc, exc_info=True)
                return await func(*args, **kwargs)
        return wrapper  # type: ignore[return-value]
    return decorator
