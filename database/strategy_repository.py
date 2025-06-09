from database.database_connection import DatabaseConnection
from validation import validate_symbol, validate_timeframe, validate_quantity, validate_risk, sanitize_input


class StrategyRepository:
    def __init__(self) -> None:
        self.db_connection = DatabaseConnection()

    async def __aenter__(self) -> "StrategyRepository":
        await self.db_connection.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
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

    async def get_strategy_by_name(self, strategy_name: str):
        strategy_name = sanitize_input(strategy_name)
        sql = """
            SELECT * FROM strategies
            WHERE strategy_name = $1
        """
        values = (strategy_name,)
        async with self.db_connection as conn:
            return await conn.execute_query(sql, values)

