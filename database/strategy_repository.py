from database.database_connection import DatabaseConnection

class StrategyRepository:
    def __init__(self):
        self.db_connection = DatabaseConnection()

    async def __aenter__(self):
        await self.db_connection.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.db_connection.disconnect()

    async def insert_strategy(self, strategy_name, strategy_type, symbol, interval, initial_balance, risk_per_trade):
        sql = """
            INSERT INTO strategies (strategy_name, strategy_type, symbol, interval, initial_balance, risk_per_trade)
            VALUES ($1, $2, $3, $4, $5, $6)
        """
        values = (strategy_name, strategy_type, symbol, interval, initial_balance, risk_per_trade)
        await self.db_connection.execute_query(sql, values)

    async def get_strategy_by_name(self, strategy_name):
        sql = """
            SELECT * FROM strategies
            WHERE strategy_name = $1
        """
        values = (strategy_name,)
        return await self.db_connection.execute_query(sql, values)