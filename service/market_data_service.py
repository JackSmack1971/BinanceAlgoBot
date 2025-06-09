import abc
import logging
from typing import Any, Dict, List

from utils import handle_error
from database.market_data_repository import MarketDataRepository
from service.repository_service import RepositoryService
from exceptions import DataError, MarketDataServiceException

logger = logging.getLogger(__name__)


class MarketDataService(abc.ABC):
    """Abstract interface for retrieving market data."""

    @abc.abstractmethod
    async def get_market_data(self, symbol: str) -> List[Dict[str, Any]]:
        """Return historical market data for ``symbol``.

        Parameters
        ----------
        symbol : str
            Trading symbol (e.g. ``"BTCUSDT"``).

        Returns
        -------
        List[Dict[str, Any]]
            A list of market data records.
        """
        ...

    @abc.abstractmethod
    async def get_latest_market_data(self, symbol: str) -> Dict[str, Any]:
        """Return the latest market data record for ``symbol``."""
        ...


class MarketDataServiceImpl(RepositoryService):
    """Concrete implementation using :class:`MarketDataRepository`."""

    def __init__(self) -> None:
        """Initialize the service and repository."""
        super().__init__()
        self.market_data_repository = MarketDataRepository()

    @handle_error
    async def get_market_data(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch historical market data from the repository."""
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
        """Fetch the most recent market data entry."""
        try:
            return await self.market_data_repository.get_market_data(symbol, page_number=1, page_size=1)
        except DataError as exc:
            logger.error("Database error in %s: %s", self.__class__.__name__, exc)
            raise MarketDataServiceException(f"Operation failed: {exc}") from exc
        except Exception as exc:
            logger.error(
                "Unexpected error getting latest market data for symbol %s: %s", symbol, exc, exc_info=True
            )
            raise MarketDataServiceException(f"Operation failed: {exc}") from exc

    async def close_connection(self) -> None:
        """Close the database connection pool."""
        await self.db_connection.disconnect()

