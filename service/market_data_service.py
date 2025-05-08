import abc
from typing import List, Dict, Any
from service.repository_service import RepositoryService, transaction
from database.market_data_repository import MarketDataRepository
from utils import handle_error
import logging

logger = logging.getLogger(__name__)

class MarketDataServiceException(BaseTradingException):
    pass

class MarketDataService(abc.ABC):
    @abc.abstractmethod
    def get_market_data(self, symbol: str) -> List[Dict[str, Any]]:
        pass

    @abc.abstractmethod
    def get_latest_market_data(self, symbol: str) -> Dict[str, Any]:
        pass

class MarketDataServiceImpl(RepositoryService):
    def __init__(self):
        super().__init__()
        self.market_data_repository = MarketDataRepository()

    @handle_error
    def get_market_data(self, symbol: str) -> List[Dict[str, Any]]:
        conn = self.get_connection()
        try:
            market_data = self.market_data_repository.get_market_data_by_symbol(symbol)
            logger.debug(f"Retrieved market data for symbol: {symbol}.")
            return market_data
        except Exception as e:
            logger.error(f"Error getting market data for symbol {symbol}: {e}", exc_info=True)
            raise DataError(f"Error getting market data for symbol {symbol}: {e}") from e
        finally:
            self.put_connection(conn)

    @handle_error
    def get_latest_market_data(self, symbol: str) -> Dict[str, Any]:
        conn = self.get_connection()
        try:
            latest_market_data = self.market_data_repository.get_latest_market_data(symbol)
            logger.debug(f"Retrieved latest market data for symbol: {symbol}.")
            return latest_market_data
        except Exception as e:
            logger.error(f"Error getting latest market data for symbol {symbol}: {e}", exc_info=True)
            raise DataError(f"Error getting latest market data for symbol {symbol}: {e}") from e
        finally:
            self.put_connection(conn)

    def close_connection(self, conn):
        conn.close()