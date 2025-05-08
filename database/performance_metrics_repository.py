from database.database_connection import DatabaseConnection

class PerformanceMetricsRepository:
    def __init__(self):
        self.db_connection = DatabaseConnection()

    async def __aenter__(self):
        await self.db_connection.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.db_connection.disconnect()

    async def insert_performance_metrics(self, trade_id, initial_capital, final_capital, total_return, annual_return, max_drawdown, sharpe_ratio, win_rate, avg_profit_pct, risk_reward_ratio, profit_factor):
        sql = """
            INSERT INTO performance_metrics (trade_id, initial_capital, final_capital, total_return, annual_return, max_drawdown, sharpe_ratio, win_rate, avg_profit_pct, risk_reward_ratio, profit_factor)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        """
        values = (trade_id, initial_capital, final_capital, total_return, annual_return, max_drawdown, sharpe_ratio, win_rate, avg_profit_pct, risk_reward_ratio, profit_factor)
        await self.db_connection.execute_query(sql, values)

    async def get_performance_metrics_by_trade_id(self, trade_id):
        sql = """
            SELECT * FROM performance_metrics
            WHERE trade_id = $1
        """
        values = (trade_id,)
        return await self.db_connection.execute_query(sql, values)