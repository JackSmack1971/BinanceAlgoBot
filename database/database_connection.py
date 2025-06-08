import asyncio
import asyncpg
import logging
from typing import Any, Iterable, Optional

from config import DATABASE_URL
from exceptions import DataError

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """Async wrapper around an asyncpg connection pool."""

    def __init__(self, min_conn: int = 1, max_conn: int = 10) -> None:
        self.conn_pool: Optional[asyncpg.pool.Pool] = None
        self.min_conn = min_conn
        self.max_conn = max_conn

    async def connect(self) -> None:
        """Initialize the async connection pool with simple retry logic."""
        attempts = 3
        for attempt in range(1, attempts + 1):
            try:
                self.conn_pool = await asyncpg.create_pool(
                    DATABASE_URL,
                    min_size=self.min_conn,
                    max_size=self.max_conn,
                )
                logger.info("Database connection pool established.")
                return
            except Exception as exc:  # pragma: no cover - initialization failures
                logger.error(
                    "Database connection attempt %s failed: %s", attempt, exc, exc_info=True
                )
                if attempt == attempts:
                    raise DataError(f"Error connecting to the database: {exc}") from exc
                await asyncio.sleep(attempt)

    async def disconnect(self) -> None:
        if self.conn_pool:
            await self.conn_pool.close()
            logger.info("Database connection pool closed.")

    async def __aenter__(self) -> "DatabaseConnection":
        if not self.conn_pool:
            await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - trivial
        # Connections are acquired per query, nothing to clean up here
        pass

    async def execute_query(
        self, query: str, params: Optional[Iterable[Any]] = None
    ) -> Any:
        """Execute a query using a pooled connection."""
        if not self.conn_pool:
            await self.connect()
        async with self.conn_pool.acquire() as conn:
            try:
                if params:
                    return await conn.fetch(query, *params)
                return await conn.fetch(query)
            except Exception as exc:
                logger.error("Error executing query: %s", exc, exc_info=True)
                raise DataError(f"Error executing query: {exc}") from exc

