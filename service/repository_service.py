import abc
import logging
from contextlib import asynccontextmanager

from database.database_connection import DatabaseConnection
from exceptions import DataError

logger = logging.getLogger(__name__)


class RepositoryService(abc.ABC):
    """Base class for services requiring database access."""

    def __init__(self) -> None:
        """Create a new repository service with its own DB connection."""
        self.db_connection = DatabaseConnection()

    @asynccontextmanager
    async def transaction(self):
        """Async context manager returning a DB transaction."""
        if not self.db_connection.conn_pool:
            await self.db_connection.connect()
        async with self.db_connection.conn_pool.acquire() as conn:
            tx = conn.transaction()
            await tx.start()
            try:
                yield conn
                await tx.commit()
                logger.debug("Transaction committed.")
            except Exception as exc:
                await tx.rollback()
                logger.error("Transaction rolled back: %s", exc, exc_info=True)
                raise DataError(f"Error committing transaction: {exc}") from exc

    @abc.abstractmethod
    async def close_connection(self) -> None:
        """Close the underlying database connection pool."""
        ...

