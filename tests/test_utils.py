import logging
import pytest
from utils import handle_error
from exceptions import BaseTradingException


@handle_error
def faulty_function():
    raise ValueError("boom")


@handle_error
async def faulty_async_function():
    raise ValueError("boom")


def test_handle_error_raises_custom_exception(caplog):
    caplog.set_level(logging.ERROR)
    with pytest.raises(BaseTradingException):
        faulty_function()
    assert any("Error in faulty_function" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_handle_error_async_raises_custom_exception(caplog):
    caplog.set_level(logging.ERROR)
    with pytest.raises(BaseTradingException):
        await faulty_async_function()
    assert any("Error in faulty_async_function" in r.message for r in caplog.records)
