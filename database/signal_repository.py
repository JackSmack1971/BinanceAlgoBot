from database.database_connection import DatabaseConnection
from validation import validate_quantity, sanitize_input


class SignalRepository:
    def __init__(self) -> None:
        self.db_connection = DatabaseConnection()

    async def __aenter__(self) -> "SignalRepository":
        await self.db_connection.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.db_connection.disconnect()

    async def insert_signal(self, market_data_id: int, strategy_id: int, signal: str) -> None:
        market_data_id = int(validate_quantity(market_data_id))
        strategy_id = int(validate_quantity(strategy_id))
        signal = sanitize_input(signal)
        sql = """
            INSERT INTO signals (market_data_id, strategy_id, signal)
            VALUES ($1, $2, $3)
        """
        values = (market_data_id, strategy_id, signal)
        async with self.db_connection as conn:
            await conn.execute_query(sql, values)

    async def get_signals_by_strategy_id(self, strategy_id: int):
        strategy_id = int(validate_quantity(strategy_id))
        sql = """
            SELECT * FROM signals
            WHERE strategy_id = $1
        """
        values = (strategy_id,)
        async with self.db_connection as conn:
            return await conn.execute_query(sql, values)

