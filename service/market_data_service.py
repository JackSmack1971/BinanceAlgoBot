import abc
import logging
from typing import Any, Dict, List

from utils import handle_error
from database.market_data_repository import MarketDataRepository
from service.repository_service import RepositoryService
from exceptions import DataError, MarketDataServiceException

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
        except DataError as exc:
            logger.error("Database error in %s: %s", self.__class__.__name__, exc)
            raise MarketDataServiceException(f"Operation failed: {exc}") from exc
        except Exception as exc:
            logger.error("Unexpected error getting market data for symbol %s: %s", symbol, exc, exc_info=True)
            raise MarketDataServiceException(f"Operation failed: {exc}") from exc

    @handle_error
    async def get_latest_market_data(self, symbol: str) -> Dict[str, Any]:
        try:
            return await self.market_data_repository.get_market_data(symbol, page_number=1, page_size=1)
        except DataError as exc:
            logger.error("Database error in %s: %s", self.__class__.__name__, exc)
            raise MarketDataServiceException(f"Operation failed: {exc}") from exc
        except Exception as exc:
            logger.error("Unexpected error getting latest market data for symbol %s: %s", symbol, exc, exc_info=True)
            raise MarketDataServiceException(f"Operation failed: {exc}") from exc

    async def close_connection(self) -> None:
        await self.db_connection.disconnect()

