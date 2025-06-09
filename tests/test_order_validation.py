import pytest
from pydantic import ValidationError

from src.execution.order_validation import validate_order


@pytest.mark.asyncio
async def test_validate_order_success():
    result = await validate_order("BTCUSDT", 1, 10)
    assert result.symbol == "BTCUSDT"


@pytest.mark.asyncio
async def test_validate_order_failure():
    with pytest.raises(ValidationError):
        await validate_order("BTCUSDT", -1, 10)
