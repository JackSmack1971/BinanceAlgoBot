from database.database_connection import DatabaseConnection

class SignalRepository:
    def __init__(self):
        self.db_connection = DatabaseConnection()

    async def __aenter__(self):
        await self.db_connection.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.db_connection.disconnect()

    async def insert_signal(self, market_data_id, strategy_id, signal):
        sql = """
            INSERT INTO signals (market_data_id, strategy_id, signal)
            VALUES ($1, $2, $3)
        """
        values = (market_data_id, strategy_id, signal)
        await self.db_connection.execute_query(sql, values)

    async def get_signals_by_strategy_id(self, strategy_id):
        sql = """
            SELECT * FROM signals
            WHERE strategy_id = $1
        """
        values = (strategy_id,)
        return await self.db_connection.execute_query(sql, values)