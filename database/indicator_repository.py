from database.database_connection import DatabaseConnection

class IndicatorRepository:
    def __init__(self):
        self.db_connection = DatabaseConnection()

    async def __aenter__(self):
        await self.db_connection.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.db_connection.disconnect()

    async def insert_indicator(self, market_data_id, ema, rsi, atr, vwap):
        sql = """
            INSERT INTO indicators (market_data_id, ema, rsi, atr, vwap)
            VALUES ($1, $2, $3, $4, $5)
        """
        values = (market_data_id, ema, rsi, atr, vwap)
        await self.db_connection.execute_query(sql, values)

    async def get_indicators_by_market_data_id(self, market_data_id):
        sql = """
            SELECT * FROM indicators
            WHERE market_data_id = $1
        """
        values = (market_data_id,)
        return await self.db_connection.execute_query(sql, values)