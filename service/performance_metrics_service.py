import abc
import logging
from typing import Any, Dict

from utils import handle_error
from database.performance_metrics_repository import PerformanceMetricsRepository
from service.repository_service import RepositoryService
from exceptions import DataError, PerformanceMetricsServiceException

logger = logging.getLogger(__name__)


class PerformanceMetricsService(abc.ABC):
    @abc.abstractmethod
    async def get_performance_metrics_by_trade_id(self, trade_id: int) -> Dict[str, Any]:
        ...

    @abc.abstractmethod
    async def insert_performance_metrics(
        self,
        trade_id: int,
        initial_capital: float,
        final_capital: float,
        total_return: float,
        annual_return: float,
        max_drawdown: float,
        sharpe_ratio: float,
        win_rate: float,
        avg_profit_pct: float,
        risk_reward_ratio: float,
        profit_factor: float,
    ) -> None:
        ...


class PerformanceMetricsServiceImpl(RepositoryService):
    def __init__(self) -> None:
        super().__init__()
        self.performance_metrics_repository = PerformanceMetricsRepository()

    @handle_error
    async def get_performance_metrics_by_trade_id(self, trade_id: int) -> Dict[str, Any]:
        try:
            return await self.performance_metrics_repository.get_performance_metrics_by_trade_id(trade_id)
        except DataError as exc:
            logger.error("Database error in %s: %s", self.__class__.__name__, exc)
            raise PerformanceMetricsServiceException(f"Operation failed: {exc}") from exc
        except Exception as exc:
            logger.error("Unexpected error getting performance metrics for trade ID %s: %s", trade_id, exc, exc_info=True)
            raise PerformanceMetricsServiceException(f"Operation failed: {exc}") from exc

    @handle_error
    async def insert_performance_metrics(
        self,
        trade_id: int,
        initial_capital: float,
        final_capital: float,
        total_return: float,
        annual_return: float,
        max_drawdown: float,
        sharpe_ratio: float,
        win_rate: float,
        avg_profit_pct: float,
        risk_reward_ratio: float,
        profit_factor: float,
    ) -> None:
        try:
            async with self.transaction():
                await self.performance_metrics_repository.insert_performance_metrics(
                    trade_id,
                    initial_capital,
                    final_capital,
                    total_return,
                    annual_return,
                    max_drawdown,
                    sharpe_ratio,
                    win_rate,
                    avg_profit_pct,
                    risk_reward_ratio,
                    profit_factor,
                )
                logger.debug("Inserted performance metrics for trade ID %s.", trade_id)
        except DataError as exc:
            logger.error("Database error in %s: %s", self.__class__.__name__, exc)
            raise PerformanceMetricsServiceException(f"Operation failed: {exc}") from exc
        except Exception as exc:
            logger.error("Unexpected error inserting performance metrics for trade ID %s: %s", trade_id, exc, exc_info=True)
            raise PerformanceMetricsServiceException(f"Operation failed: {exc}") from exc

    async def close_connection(self) -> None:
        await self.db_connection.disconnect()

