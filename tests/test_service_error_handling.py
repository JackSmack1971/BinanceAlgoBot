import pytest

from service.market_data_service import MarketDataServiceImpl
from exceptions import MarketDataServiceException


@pytest.mark.asyncio
async def test_market_data_service_error(monkeypatch):
    service = MarketDataServiceImpl()

    async def fail(*args, **kwargs):
        raise Exception("db fail")

    service.market_data_repository.get_market_data = fail  # type: ignore

    with pytest.raises(MarketDataServiceException):
        await service.get_market_data("BTCUSDT")
