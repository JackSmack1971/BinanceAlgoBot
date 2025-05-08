import abc
from typing import List, Dict, Any
from service.repository_service import RepositoryService, transaction
from database.indicator_repository import IndicatorRepository
from database.market_data_repository import MarketDataRepository
from utils import handle_error
import logging

logger = logging.getLogger(__name__)

class IndicatorServiceException(BaseTradingException):
    pass

class IndicatorService(abc.ABC):
    @abc.abstractmethod
    def calculate_indicators(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pass

    @abc.abstractmethod
    def get_indicators(self, indicator_name: str) -> List[Dict[str, Any]]:
        pass

class IndicatorServiceImpl(RepositoryService):
    def __init__(self):
        super().__init__()
        self.indicator_repository = IndicatorRepository()
        self.market_data_repository = MarketDataRepository()

    @handle_error
    def calculate_indicators(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        conn = self.get_connection()
        try:
            with self.transaction(conn):
                for item in data:
                    # Assuming indicator calculation logic here
                    # Example: item['indicator_value'] = calculate_some_indicator(item['close'])
                    self.indicator_repository.insert_indicator(
                        timestamp=item['timestamp'],
                        indicator_name='some_indicator',
                        indicator_value=item['indicator_value']
                    )
                logger.debug("Calculated and inserted indicators.")
                return data
        except Exception as e:
            logger.error(f"Error calculating and inserting indicators: {e}", exc_info=True)
            raise DataError(f"Error calculating and inserting indicators: {e}") from e
        finally:
            self.put_connection(conn)

    @handle_error
    def get_indicators(self, indicator_name: str) -> List[Dict[str, Any]]:
        conn = self.get_connection()
        try:
            indicators = self.indicator_repository.get_indicators_by_name(indicator_name)
            logger.debug(f"Retrieved indicators by name: {indicator_name}.")
            return indicators
        except Exception as e:
            logger.error(f"Error getting indicators by name {indicator_name}: {e}", exc_info=True)
            raise DataError(f"Error getting indicators by name {indicator_name}: {e}") from e
        finally:
            self.put_connection(conn)

    def close_connection(self, conn):
        conn.close()