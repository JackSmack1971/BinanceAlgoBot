from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum, auto
from typing import Any, Awaitable, Callable, TypeVar


class CircuitBreakerState(Enum):
    CLOSED = auto()
    OPEN = auto()
    HALF_OPEN = auto()


class CircuitBreakerError(Exception):
    """Raised when the circuit is open and calls are blocked."""


T = TypeVar("T")


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int,
        recovery_timeout: float,
        success_threshold: int,
        logger: logging.Logger | None = None,
    ) -> None:
        if failure_threshold <= 0 or recovery_timeout <= 0 or success_threshold <= 0:
            raise ValueError("Invalid circuit breaker configuration")
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._opened_at = 0.0
        self.logger = logger or logging.getLogger(__name__)
        self._configure_logger()

    def _configure_logger(self) -> None:
        fmt = "%(asctime)s.%(msecs)03d %(name)s %(levelname)s %(message)s"
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt, "%Y-%m-%d %H:%M:%S"))
        if not self.logger.handlers:
            self.logger.addHandler(handler)
        self.logger.propagate = False

    def _log_state(self, state: CircuitBreakerState) -> None:
        self.logger.info("Circuit breaker state -> %s", state.name)

    def _change_state(self, state: CircuitBreakerState) -> None:
        self._state = state
        if state is CircuitBreakerState.OPEN:
            self._opened_at = time.monotonic()
        self._failure_count = 0
        self._success_count = 0
        self._log_state(state)

    def _allow(self) -> bool:
        if self._state is CircuitBreakerState.OPEN and time.monotonic() - self._opened_at >= self.recovery_timeout:
            self._change_state(CircuitBreakerState.HALF_OPEN)
        return self._state is not CircuitBreakerState.OPEN

    def record_success(self) -> None:
        if self._state is CircuitBreakerState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.success_threshold:
                self._change_state(CircuitBreakerState.CLOSED)
        else:
            self._failure_count = 0

    def record_failure(self) -> None:
        self._failure_count += 1
        if self._failure_count >= self.failure_threshold:
            self._change_state(CircuitBreakerState.OPEN)

    async def call(self, func: Callable[..., Awaitable[T]], *args: Any, timeout: float = 5.0, **kwargs: Any) -> T:
        if not self._allow():
            raise CircuitBreakerError("Circuit breaker is open")
        try:
            result = await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
            self.record_success()
            return result
        except Exception:
            self.record_failure()
            raise
