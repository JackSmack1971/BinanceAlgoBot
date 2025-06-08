import abc
from typing import Dict, Any
from service.repository_service import RepositoryService, transaction
from database.performance_metrics_repository import PerformanceMetricsRepository
from utils import handle_error
import logging
from exceptions import BaseTradingException, DataError, PerformanceMetricsServiceException

logger = logging.getLogger(__name__)

class PerformanceMetricsService(abc.ABC):
    @abc.abstractmethod
    def get_performance_metrics_by_trade_id(self, trade_id: int) -> Dict[str, Any]:
        pass

    @abc.abstractmethod
    def insert_performance_metrics(self, trade_id: int, initial_capital: float, final_capital: float, total_return: float, annual_return: float, max_drawdown: float, sharpe_ratio: float, win_rate: float, avg_profit_pct: float, risk_reward_ratio: float, profit_factor: float):
        pass

class PerformanceMetricsServiceImpl(RepositoryService):
    def __init__(self):
        super().__init__()
        self.performance_metrics_repository = PerformanceMetricsRepository()

    @handle_error
    def get_performance_metrics_by_trade_id(self, trade_id: int) -> Dict[str, Any]:
        conn = self.get_connection()
        try:
            metrics = self.performance_metrics_repository.get_performance_metrics_by_trade_id(trade_id)
            logger.debug(f"Retrieved performance metrics for trade ID {trade_id}.")
            return metrics
        except Exception as e:
            logger.error(f"Error getting performance metrics for trade ID {trade_id}: {e}", exc_info=True)
            raise DataError(f"Error getting performance metrics for trade ID {trade_id}: {e}") from e
        finally:
            self.put_connection(conn)

    @handle_error
    def insert_performance_metrics(self, trade_id: int, initial_capital: float, final_capital: float, total_return: float, annual_return: float, max_drawdown: float, sharpe_ratio: float, win_rate: float, avg_profit_pct: float, risk_reward_ratio: float, profit_factor: float):
        conn = self.get_connection()
        try:
            with self.transaction(conn):
                self.performance_metrics_repository.insert_performance_metrics(trade_id, initial_capital, final_capital, total_return, annual_return, max_drawdown, sharpe_ratio, win_rate, avg_profit_pct, risk_reward_ratio, profit_factor)
                logger.debug(f"Inserted performance metrics for trade ID {trade_id}.")
        except Exception as e:
            logger.error(f"Error inserting performance metrics for trade ID {trade_id}: {e}", exc_info=True)
            raise DataError(f"Error inserting performance metrics for trade ID {trade_id}: {e}") from e
        finally:
            self.put_connection(conn)

    def close_connection(self, conn):
        conn.close()