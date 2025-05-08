from database.database_connection import DatabaseConnection

class TradeHistoryRepository:
    def __init__(self):
        self.db_connection = DatabaseConnection()

    async def __aenter__(self):
        await self.db_connection.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.db_connection.disconnect()

    async def insert_trade_history(self, strategy_id, entry_time, exit_time, position_type, entry_price, exit_price, profit_pct, duration, commission_fee):
        sql = """
            INSERT INTO trade_history (strategy_id, entry_time, exit_time, position_type, entry_price, exit_price, profit_pct, duration, commission_fee)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """
        values = (strategy_id, entry_time, exit_time, position_type, entry_price, exit_price, profit_pct, duration, commission_fee)
        await self.db_connection.execute_query(sql, values)

    async def get_trade_history_by_strategy_id(self, strategy_id):
        sql = """
            SELECT * FROM trade_history
            WHERE strategy_id = $1
        """
        values = (strategy_id,)
        return await self.db_connection.execute_query(sql, values)