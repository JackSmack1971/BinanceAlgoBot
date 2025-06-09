from __future__ import annotations

import asyncio
import hashlib
import json
import os
import pickle
import time
import zlib
from io import BytesIO
from typing import Any, Awaitable, Callable, Iterable

import numpy as np
import redis.asyncio as aioredis


class CacheError(Exception):
    """Base class for cache errors."""


class CircuitBreakerOpen(CacheError):
    """Raised when the circuit breaker is open."""


def _serialize(obj: Any) -> bytes:
    buf = BytesIO()
    if isinstance(obj, np.ndarray):
        np.save(buf, obj, allow_pickle=False)
        marker = b"n"
    else:
        buf.write(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
        marker = b"p"
    payload = marker + buf.getvalue()
    if len(payload) > 1024:
        return b"1" + zlib.compress(payload)
    return b"0" + payload


def _deserialize(raw: bytes) -> Any:
    if not raw:
        return None
    flag, payload = raw[0:1], raw[1:]
    if flag == b"1":
        payload = zlib.decompress(payload)
    marker, data = payload[0:1], payload[1:]
    if marker == b"n":
        return np.load(BytesIO(data), allow_pickle=False)
    return pickle.loads(data)


def _hash(params: Any) -> str:
    try:
        encoded = json.dumps(params, sort_keys=True, default=str).encode()
    except TypeError:
        encoded = pickle.dumps(params)
    return hashlib.sha256(encoded).hexdigest()


class IntelligentRedisCache:
    """Advanced Redis cache with compression and fault tolerance."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        db: int | None = None,
        failure_threshold: int = 3,
        open_timeout: int = 30,
    ) -> None:
        self.client = aioredis.Redis(
            host=host or os.getenv("REDIS_HOST", "localhost"),
            port=int(port or os.getenv("REDIS_PORT", "6379")),
            db=int(db or os.getenv("REDIS_DB", "0")),
        )
        self.failure_threshold = failure_threshold
        self.open_timeout = open_timeout
        self._failures = 0
        self._open_until = 0.0
        self.hits = 0
        self.misses = 0

    def _is_open(self) -> bool:
        return time.time() < self._open_until

    async def _record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._open_until = time.time() + self.open_timeout
            self._failures = 0

    def build_key(self, namespace: str, params: Any) -> str:
        return f"{namespace}:{_hash(params)}"

    async def get(self, namespace: str, params: Any) -> Any | None:
        if self._is_open():
            raise CircuitBreakerOpen("cache unavailable")
        key = self.build_key(namespace, params)
        try:
            raw = await asyncio.wait_for(self.client.get(key), timeout=1.0)
            if raw is None:
                self.misses += 1
                return None
            self.hits += 1
            return _deserialize(raw)
        except Exception:
            await self._record_failure()
            return None

    async def set(self, namespace: str, params: Any, value: Any, ttl: int) -> None:
        if self._is_open():
            return
        key = self.build_key(namespace, params)
        try:
            payload = _serialize(value)
            await asyncio.wait_for(self.client.setex(key, ttl, payload), timeout=1.0)
        except Exception:
            await self._record_failure()

    async def invalidate(self, prefix: str) -> None:
        if self._is_open():
            return
        try:
            keys = await asyncio.wait_for(self.client.keys(f"{prefix}*"), timeout=1.0)
            if keys:
                await self.client.delete(*keys)
        except Exception:
            await self._record_failure()

    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total) if total else 0.0

    async def warm_cache(
        self,
        tasks: Iterable[tuple[str, Callable[..., Awaitable[Any]], tuple, dict, int]],
    ) -> None:
        for ns, func, args, kwargs, ttl in tasks:
            result = await func(*args, **kwargs)
            await self.set(ns, {"args": args, "kwargs": kwargs}, result, ttl)


F = Callable[..., Awaitable[Any]]
_default_cache = IntelligentRedisCache()


def cache_function(namespace: str, ttl: int = 300) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            params = {"args": args, "kwargs": kwargs}
            cached = await _default_cache.get(namespace, params)
            if cached is not None:
                return cached
            result = await func(*args, **kwargs)
            await _default_cache.set(namespace, params, result, ttl)
            return result

        return wrapper  # type: ignore[return-value]

    return decorator
