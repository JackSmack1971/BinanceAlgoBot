import asyncpg
import logging
from config import DATABASE_URL
from exceptions import DataError

logger = logging.getLogger(__name__)

class DatabaseConnection:
    def __init__(self, min_conn=1, max_conn=10):
        self.conn_pool = None
        self.min_conn = min_conn
        self.max_conn = max_conn

    async def connect(self):
        try:
            self.conn_pool = await asyncpg.create_pool(DATABASE_URL, min_size=self.min_conn, max_size=self.max_conn)
            print("Database connection pool established.")
        except Exception as e:
            logger.error(f"Error connecting to the database: {e}", exc_info=True)
            raise DataError(f"Error connecting to the database: {e}") from e

    async def disconnect(self):
        if self.conn_pool:
            await self.conn_pool.close()
            print("Database connection pool closed.")

    async def get_connection(self):
        if not self.conn_pool:
            await self.connect()
        return await self.conn_pool.acquire()

    async def release_connection(self, conn):
        if self.conn_pool:
            await self.conn_pool.release(conn)

    async def execute_query(self, query, params=None):
        conn = None
        try:
            conn = await self.get_connection()
            result = await conn.fetch(query, *params) if params else await conn.fetch(query)
            return result
        except Exception as e:
            logger.error(f"Error executing query: {e}", exc_info=True)
            raise DataError(f"Error executing query: {e}") from e
            raise
        finally:
            if conn:
                await self.release_connection(conn)