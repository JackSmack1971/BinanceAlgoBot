import abc
import logging
from typing import Any, Dict, List

from utils import handle_error
from database.indicator_repository import IndicatorRepository
from database.market_data_repository import MarketDataRepository
from service.repository_service import RepositoryService
from exceptions import DataError, IndicatorServiceException

logger = logging.getLogger(__name__)


class IndicatorService(abc.ABC):
    """Abstract interface for indicator calculations."""

    @abc.abstractmethod
    async def calculate_indicators(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Persist calculated indicators for provided market data."""
        ...

    @abc.abstractmethod
    async def get_indicators(self, market_data_id: int) -> List[Dict[str, Any]]:
        """Retrieve indicators for a specific market data record."""
        ...


class IndicatorServiceImpl(RepositoryService):
    """Concrete implementation storing indicators in the database."""

    def __init__(self) -> None:
        """Initialize repositories used by the service."""
        super().__init__()
        self.indicator_repository = IndicatorRepository()
        self.market_data_repository = MarketDataRepository()

    @handle_error
    async def calculate_indicators(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate and store indicators for the provided data list."""
        try:
            async with self.transaction():
                for item in data:
                    await self.indicator_repository.insert_indicator(
                        market_data_id=item["market_data_id"],
                        ema=item["ema"],
                        rsi=item["rsi"],
                        atr=item["atr"],
                        vwap=item["vwap"],
                    )
                logger.debug("Calculated and inserted indicators.")
            return data
        except DataError as exc:
            logger.error("Database error in %s: %s", self.__class__.__name__, exc)
            raise IndicatorServiceException(f"Operation failed: {exc}") from exc
        except Exception as exc:
            logger.error("Unexpected error calculating indicators: %s", exc, exc_info=True)
            raise IndicatorServiceException(f"Operation failed: {exc}") from exc

    @handle_error
    async def get_indicators(self, market_data_id: int) -> List[Dict[str, Any]]:
        """Get calculated indicators for a given market data ID."""
        try:
            return await self.indicator_repository.get_indicators_by_market_data_id(market_data_id)
        except DataError as exc:
            logger.error("Database error in %s: %s", self.__class__.__name__, exc)
            raise IndicatorServiceException(f"Operation failed: {exc}") from exc
        except Exception as exc:
            logger.error("Unexpected error getting indicators for id %s: %s", market_data_id, exc, exc_info=True)
            raise IndicatorServiceException(f"Operation failed: {exc}") from exc

    async def close_connection(self) -> None:
        """Close the database connection pool."""
        await self.db_connection.disconnect()

