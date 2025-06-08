import abc
import logging
from typing import Any, Dict, List

from utils import handle_error
from database.market_data_repository import MarketDataRepository
from service.repository_service import RepositoryService
from exceptions import DataError

logger = logging.getLogger(__name__)


class MarketDataService(abc.ABC):
    @abc.abstractmethod
    async def get_market_data(self, symbol: str) -> List[Dict[str, Any]]:
        ...

    @abc.abstractmethod
    async def get_latest_market_data(self, symbol: str) -> Dict[str, Any]:
        ...


class MarketDataServiceImpl(RepositoryService):
    def __init__(self) -> None:
        super().__init__()
        self.market_data_repository = MarketDataRepository()

    @handle_error
    async def get_market_data(self, symbol: str) -> List[Dict[str, Any]]:
        try:
            return await self.market_data_repository.get_market_data(symbol)
        except Exception as exc:
            logger.error("Error getting market data for symbol %s: %s", symbol, exc, exc_info=True)
            raise DataError(f"Error getting market data for symbol {symbol}: {exc}") from exc

    @handle_error
    async def get_latest_market_data(self, symbol: str) -> Dict[str, Any]:
        try:
            return await self.market_data_repository.get_market_data(symbol, page_number=1, page_size=1)
        except Exception as exc:
            logger.error("Error getting latest market data for symbol %s: %s", symbol, exc, exc_info=True)
            raise DataError(f"Error getting latest market data for symbol {symbol}: {exc}") from exc

    async def close_connection(self) -> None:
        await self.db_connection.disconnect()

