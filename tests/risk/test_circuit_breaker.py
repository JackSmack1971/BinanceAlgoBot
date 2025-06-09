import asyncio
import pytest

from src.risk.circuit_breaker import CircuitBreaker, CircuitBreakerError, CircuitBreakerState


@pytest.mark.asyncio
async def test_circuit_breaker_state_transitions() -> None:
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1, success_threshold=1)

    async def fail() -> None:
        raise ValueError("boom")

    with pytest.raises(ValueError):
        await cb.call(fail)
    assert cb._state is CircuitBreakerState.OPEN

    await asyncio.sleep(0.15)

    async def succeed() -> int:
        return 42

    result = await cb.call(succeed)
    assert result == 42
    assert cb._state is CircuitBreakerState.CLOSED


@pytest.mark.asyncio
async def test_circuit_breaker_blocks_calls_when_open() -> None:
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1.0, success_threshold=1)

    async def fail() -> None:
        raise ValueError()

    with pytest.raises(ValueError):
        await cb.call(fail)

    async def succeed() -> None:
        return None

    with pytest.raises(CircuitBreakerError):
        await cb.call(succeed)
