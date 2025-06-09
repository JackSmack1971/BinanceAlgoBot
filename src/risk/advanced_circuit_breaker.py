import asyncio
import time
from typing import Any, Callable, Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class FailureType(Enum):
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    CONNECTION_ERROR = "connection_error"
    HTTP_ERROR = "http_error"
    AUTHENTICATION_ERROR = "auth_error"
    UNKNOWN = "unknown"


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3
    request_timeout: float = 10.0
    slow_request_threshold: float = 5.0
    max_requests_per_minute: int = 60
    rate_limit_window: float = 60.0
    monitor_window: float = 300.0
    failure_rate_threshold: float = 0.5
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    backoff_multiplier: float = 2.0


@dataclass
class RequestMetrics:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeouts: int = 0
    rate_limits: int = 0
    total_response_time: float = 0.0
    min_response_time: float = float("inf")
    max_response_time: float = 0.0
    recent_requests: List[tuple[datetime, bool, float]] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests

    @property
    def average_response_time(self) -> float:
        if self.successful_requests == 0:
            return 0.0
        return self.total_response_time / self.successful_requests

    def add_request(self, success: bool, duration: float) -> None:
        self.total_requests += 1
        if success:
            self.successful_requests += 1
            self.total_response_time += duration
            self.min_response_time = min(self.min_response_time, duration)
            self.max_response_time = max(self.max_response_time, duration)
        else:
            self.failed_requests += 1
        self.recent_requests.append((datetime.utcnow(), success, duration))
        cutoff = datetime.utcnow() - timedelta(minutes=5)
        self.recent_requests = [
            (ts, s, d) for ts, s, d in self.recent_requests if ts > cutoff
        ]


class CircuitBreakerError(Exception):
    pass


class RateLimitError(Exception):
    pass


class AdvancedCircuitBreaker:
    def __init__(self, name: str, config: CircuitBreakerConfig) -> None:
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.metrics = RequestMetrics()
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state_change_time = datetime.utcnow()
        self.request_times: List[datetime] = []
        self._lock = asyncio.Lock()
        self.logger = logging.getLogger(f"circuit_breaker.{name}")
        self.on_state_change: Optional[Callable] = None
        self.on_failure: Optional[Callable] = None
        self.on_recovery: Optional[Callable] = None

    async def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        async with self._lock:
            await self._update_state()
            if self.state == CircuitState.OPEN:
                raise CircuitBreakerError(f"Circuit breaker {self.name} is OPEN")
            await self._check_rate_limit()
        return await self._execute_with_monitoring(func, *args, **kwargs)

    async def _execute_with_monitoring(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        start = time.perf_counter()
        success = False
        try:
            result = await asyncio.wait_for(
                func(*args, **kwargs), timeout=self.config.request_timeout
            )
            success = True
            duration = time.perf_counter() - start
            await self._record_success(duration)
            return result
        except asyncio.TimeoutError as exc:
            duration = time.perf_counter() - start
            await self._record_failure(FailureType.TIMEOUT, exc, duration)
            raise
        except Exception as exc:
            duration = time.perf_counter() - start
            failure_type = self._classify_error(exc)
            await self._record_failure(failure_type, exc, duration)
            raise
        finally:
            self.metrics.add_request(success, time.perf_counter() - start)

    async def _record_success(self, duration: float) -> None:
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    await self._transition_to_closed()
            elif self.state == CircuitState.CLOSED:
                self.failure_count = max(0, self.failure_count - 1)
                if duration > self.config.slow_request_threshold:
                    self.logger.warning("Slow request: %.2fs", duration)

    async def _record_failure(
        self, failure_type: FailureType, error: Exception, duration: float
    ) -> None:
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.utcnow()
            if failure_type == FailureType.TIMEOUT:
                self.metrics.timeouts += 1
            elif failure_type == FailureType.RATE_LIMIT:
                self.metrics.rate_limits += 1
            if (
                self.state == CircuitState.CLOSED
                and self.failure_count >= self.config.failure_threshold
            ):
                await self._transition_to_open()
            elif self.state == CircuitState.HALF_OPEN:
                await self._transition_to_open()
            self.logger.error("Request failed: %s - %s", failure_type.value, error)
            if self.on_failure:
                try:
                    await self.on_failure(failure_type, error, self.metrics)
                except Exception as callback_error:
                    self.logger.error("Failure callback error: %s", callback_error)

    def _classify_error(self, error: Exception) -> FailureType:
        msg = str(error).lower()
        if isinstance(error, asyncio.TimeoutError):
            return FailureType.TIMEOUT
        if "rate limit" in msg or "429" in msg:
            return FailureType.RATE_LIMIT
        if "connection" in msg or "network" in msg:
            return FailureType.CONNECTION_ERROR
        if any(code in msg for code in ["400", "404", "500", "502", "503", "504"]):
            return FailureType.HTTP_ERROR
        if any(code in msg for code in ["401", "403", "auth"]):
            return FailureType.AUTHENTICATION_ERROR
        return FailureType.UNKNOWN

    async def _check_rate_limit(self) -> None:
        now = datetime.utcnow()
        cutoff = now - timedelta(seconds=self.config.rate_limit_window)
        self.request_times = [t for t in self.request_times if t > cutoff]
        if len(self.request_times) >= self.config.max_requests_per_minute:
            raise RateLimitError("Rate limit exceeded")
        self.request_times.append(now)

    async def _update_state(self) -> None:
        if self.state == CircuitState.OPEN and self.last_failure_time:
            elapsed = (datetime.utcnow() - self.last_failure_time).total_seconds()
            if elapsed >= self.config.recovery_timeout:
                await self._transition_to_half_open()
        elif self.state == CircuitState.CLOSED:
            await self._check_failure_rate()

    async def _check_failure_rate(self) -> None:
        if not self.metrics.recent_requests:
            return
        cutoff = datetime.utcnow() - timedelta(seconds=self.config.monitor_window)
        recent = [r for r in self.metrics.recent_requests if r[0] > cutoff]
        if len(recent) < 10:
            return
        rate = sum(1 for _, s, _ in recent if not s) / len(recent)
        if rate > self.config.failure_rate_threshold:
            self.logger.warning("High failure rate: %.1f%%", rate * 100)
            await self._transition_to_open()

    async def _transition_to_open(self) -> None:
        if self.state != CircuitState.OPEN:
            self.state = CircuitState.OPEN
            self.state_change_time = datetime.utcnow()
            self.success_count = 0
            self.logger.error("Circuit breaker %s OPENED", self.name)
            if self.on_state_change:
                await self.on_state_change(CircuitState.OPEN, self.metrics)

    async def _transition_to_half_open(self) -> None:
        self.state = CircuitState.HALF_OPEN
        self.state_change_time = datetime.utcnow()
        self.success_count = 0
        self.logger.info("Circuit breaker %s HALF-OPEN", self.name)
        if self.on_state_change:
            await self.on_state_change(CircuitState.HALF_OPEN, self.metrics)

    async def _transition_to_closed(self) -> None:
        self.state = CircuitState.CLOSED
        self.state_change_time = datetime.utcnow()
        self.failure_count = 0
        self.success_count = 0
        self.logger.info("Circuit breaker %s CLOSED", self.name)
        if self.on_recovery:
            await self.on_recovery(self.metrics)
        if self.on_state_change:
            await self.on_state_change(CircuitState.CLOSED, self.metrics)

    def get_status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "state_duration": (
                datetime.utcnow() - self.state_change_time
            ).total_seconds(),
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "success_rate": self.metrics.success_rate,
                "average_response_time": self.metrics.average_response_time,
                "timeouts": self.metrics.timeouts,
                "rate_limits": self.metrics.rate_limits,
            },
        }


class CircuitBreakerRegistry:
    def __init__(self) -> None:
        self.breakers: Dict[str, AdvancedCircuitBreaker] = {}
        self.logger = logging.getLogger("circuit_breaker_registry")

    def create_breaker(
        self, name: str, config: Optional[CircuitBreakerConfig] = None
    ) -> AdvancedCircuitBreaker:
        if name in self.breakers:
            return self.breakers[name]
        config = config or CircuitBreakerConfig()
        breaker = AdvancedCircuitBreaker(name, config)
        self.breakers[name] = breaker
        self.logger.info("Created circuit breaker: %s", name)
        return breaker

    def get_breaker(self, name: str) -> Optional[AdvancedCircuitBreaker]:
        return self.breakers.get(name)

    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        return {name: br.get_status() for name, br in self.breakers.items()}


class CircuitBreakerMonitor:
    def __init__(self, registry: CircuitBreakerRegistry) -> None:
        self.registry = registry
        self.logger = logging.getLogger("circuit_breaker_monitor")
        self._task: Optional[asyncio.Task[Any]] = None

    async def start(self, interval: float = 30.0) -> None:
        self._task = asyncio.create_task(self._loop(interval))

    async def _loop(self, interval: float) -> None:
        while True:
            try:
                await self._check()
                await asyncio.sleep(interval)
            except Exception as exc:  # pragma: no cover - monitoring should not crash
                self.logger.error("Monitor error: %s", exc)
                await asyncio.sleep(interval)

    async def _check(self) -> None:
        for name, status in self.registry.get_all_status().items():
            if status["state"] == "open":
                self.logger.error("ALERT: circuit %s OPEN", name)
            elif status["metrics"]["success_rate"] < 0.8:
                self.logger.warning(
                    "ALERT: circuit %s failure rate high %.1f%%",
                    name,
                    (1 - status["metrics"]["success_rate"]) * 100,
                )
