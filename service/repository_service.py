import abc
from database.database_connection import DatabaseConnection
import psycopg2
import psycopg2.pool
from contextlib import contextmanager
import logging
from exceptions import DataError

logger = logging.getLogger(__name__)

class RepositoryService(abc.ABC):
    def __init__(self, min_connections=1, max_connections=10, connection_timeout=5):
        self.db_connection = DatabaseConnection()
        try:
            self.connection_pool = psycopg2.pool.SimpleConnectionPool(
                minconn=min_connections,
                maxconn=max_connections,
                host=self.db_connection.host,
                port=self.db_connection.port,
                database=self.db_connection.database,
                user=self.db_connection.user,
                password=self.db_connection.password,
                connect_timeout=connection_timeout
            )
            logger.info("Connection pool initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing connection pool: {e}", exc_info=True)
            raise DataError(f"Error initializing connection pool: {e}") from e
            raise

    def get_connection(self):
        try:
            conn = self.connection_pool.getconn()
            logger.debug("Connection retrieved from pool.")
            return conn
        except Exception as e:
            logger.error(f"Error getting connection from pool: {e}", exc_info=True)
            raise DataError(f"Error getting connection from pool: {e}") from e
            raise

    def put_connection(self, conn):
        try:
            self.connection_pool.putconn(conn)
            logger.debug("Connection returned to pool.")
        except Exception as e:
            logger.error(f"Error returning connection to pool: {e}", exc_info=True)
            raise DataError(f"Error returning connection to pool: {e}") from e

    @contextmanager
    def transaction(self, conn):
        try:
            conn.begin()
            yield
            conn.commit()
            logger.debug("Transaction committed.")
        except Exception as e:
            conn.rollback()
            logger.error(f"Error committing transaction: {e}", exc_info=True)
            raise DataError(f"Error committing transaction: {e}") from e
            logger.error(f"Transaction rolled back: {e}", exc_info=True)
            raise
        finally:
            pass

    @abc.abstractmethod
    def close_connection(self, conn):
        pass