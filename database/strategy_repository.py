from database.database_connection import DatabaseConnection
from validation import validate_symbol, validate_timeframe, validate_quantity, validate_risk, sanitize_input


class StrategyRepository:
    """CRUD operations for strategy records."""

    def __init__(self) -> None:
        """Instantiate with a ``DatabaseConnection``."""
        self.db_connection = DatabaseConnection()

    async def __aenter__(self) -> "StrategyRepository":
        """Enter async context by connecting to the database."""
        await self.db_connection.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Disconnect on context exit."""
        await self.db_connection.disconnect()

    async def insert_strategy(
        self,
        strategy_name: str,
        strategy_type: str,
        symbol: str,
        interval: str,
        initial_balance: float,
        risk_per_trade: float,
    ) -> None:
        """Insert a new strategy configuration into the database."""
        strategy_name = sanitize_input(strategy_name)
        strategy_type = sanitize_input(strategy_type)
        symbol = validate_symbol(symbol)
        interval = validate_timeframe(interval)
        initial_balance = validate_quantity(initial_balance)
        risk_per_trade = validate_risk(risk_per_trade)
        sql = """
            INSERT INTO strategies (
                strategy_name, strategy_type, symbol, interval,
                initial_balance, risk_per_trade
            ) VALUES ($1, $2, $3, $4, $5, $6)
        """
        values = (
            strategy_name,
            strategy_type,
            symbol,
            interval,
            initial_balance,
            risk_per_trade,
        )
        async with self.db_connection as conn:
            await conn.execute_query(sql, values)

    async def get_strategy_by_name(
        self, strategy_name: str, page_number: int = 1, page_size: int = 100
    ):
        """Retrieve strategies filtered by name with pagination."""
        strategy_name = sanitize_input(strategy_name)
        offset = (page_number - 1) * page_size
        sql = """
            SELECT * FROM strategies
            WHERE strategy_name = $1
            LIMIT $2 OFFSET $3
        """
        values = (strategy_name, page_size, offset)
        async with self.db_connection as conn:
            return await conn.execute_query(sql, values)

