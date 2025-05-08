from database.database_connection import DatabaseConnection

class BacktestResultsRepository:
    def __init__(self):
        self.db_connection = DatabaseConnection()

    async def __aenter__(self):
        await self.db_connection.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.db_connection.disconnect()

    async def insert_backtest_results(self, strategy_id, start_date, end_date, initial_capital, final_capital, total_return, annual_return, max_drawdown, sharpe_ratio, win_rate, avg_profit_pct, risk_reward_ratio, profit_factor, total_trades, winning_trades, losing_trades):
        sql = """
            INSERT INTO backtest_results (strategy_id, start_date, end_date, initial_capital, final_capital, total_return, annual_return, max_drawdown, sharpe_ratio, win_rate, avg_profit_pct, risk_reward_ratio, profit_factor, total_trades, winning_trades, losing_trades)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
        """
        values = (strategy_id, start_date, end_date, initial_capital, final_capital, total_return, annual_return, max_drawdown, sharpe_ratio, win_rate, avg_profit_pct, risk_reward_ratio, profit_factor, total_trades, winning_trades, losing_trades)
        await self.db_connection.execute_query(sql, values)

    async def get_backtest_results_by_strategy_id(self, strategy_id):
        sql = """
            SELECT * FROM backtest_results
            WHERE strategy_id = $1
        """
        values = (strategy_id,)
        return await self.db_connection.execute_query(sql, values)