import asyncio
import pytest

from src.risk.advanced_circuit_breaker import (
    AdvancedCircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
)


@pytest.mark.asyncio
async def test_advanced_circuit_breaker_opens() -> None:
    cb = AdvancedCircuitBreaker(
        "t",
        CircuitBreakerConfig(
            failure_threshold=1, recovery_timeout=0.1, success_threshold=1
        ),
    )

    async def fail() -> None:
        raise ValueError()

    with pytest.raises(ValueError):
        await cb.call(fail)
    assert cb.state.name == "OPEN"

    await asyncio.sleep(0.15)

    async def succeed() -> int:
        return 1

    result = await cb.call(succeed)
    assert result == 1
    assert cb.state.name == "CLOSED"


@pytest.mark.asyncio
async def test_advanced_circuit_breaker_blocks_open() -> None:
    cb = AdvancedCircuitBreaker(
        "t",
        CircuitBreakerConfig(
            failure_threshold=1, recovery_timeout=1.0, success_threshold=1
        ),
    )

    async def fail() -> None:
        raise ValueError()

    with pytest.raises(ValueError):
        await cb.call(fail)

    async def succeed() -> None:
        return None

    with pytest.raises(CircuitBreakerError):
        await cb.call(succeed)
