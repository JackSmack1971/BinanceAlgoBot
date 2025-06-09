from database.database_connection import DatabaseConnection
from validation import validate_quantity


class IndicatorRepository:
    def __init__(self) -> None:
        self.db_connection = DatabaseConnection()

    async def __aenter__(self) -> "IndicatorRepository":
        await self.db_connection.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.db_connection.disconnect()

    async def insert_indicator(
        self,
        market_data_id: int,
        ema: float,
        rsi: float,
        atr: float,
        vwap: float,
    ) -> None:
        market_data_id = int(validate_quantity(market_data_id))
        ema = validate_quantity(ema)
        rsi = validate_quantity(rsi)
        atr = validate_quantity(atr)
        vwap = validate_quantity(vwap)
        sql = """
            INSERT INTO indicators (market_data_id, ema, rsi, atr, vwap)
            VALUES ($1, $2, $3, $4, $5)
        """
        values = (market_data_id, ema, rsi, atr, vwap)
        async with self.db_connection as conn:
            await conn.execute_query(sql, values)

    async def get_indicators_by_market_data_id(self, market_data_id: int):
        market_data_id = int(validate_quantity(market_data_id))
        sql = """
            SELECT * FROM indicators
            WHERE market_data_id = $1
        """
        values = (market_data_id,)
        async with self.db_connection as conn:
            return await conn.execute_query(sql, values)

