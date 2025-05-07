from database.database_connection import DatabaseConnection

class BacktestResultsRepository:
    def __init__(self):
        self.db_connection = DatabaseConnection()

    def insert_backtest_results(self, strategy_id, start_date, end_date, initial_capital, final_capital, total_return, annual_return, max_drawdown, sharpe_ratio, win_rate, avg_profit_pct, risk_reward_ratio, profit_factor, total_trades, winning_trades, losing_trades):
        conn = self.db_connection.get_connection()
        cursor = conn.cursor()
        sql = """
            INSERT INTO backtest_results (strategy_id, start_date, end_date, initial_capital, final_capital, total_return, annual_return, max_drawdown, sharpe_ratio, win_rate, avg_profit_pct, risk_reward_ratio, profit_factor, total_trades, winning_trades, losing_trades)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = (strategy_id, start_date, end_date, initial_capital, final_capital, total_return, annual_return, max_drawdown, sharpe_ratio, win_rate, avg_profit_pct, risk_reward_ratio, profit_factor, total_trades, winning_trades, losing_trades)
        cursor.execute(sql, values)
        conn.commit()
        cursor.close()

    def get_backtest_results_by_strategy_id(self, strategy_id):
        conn = self.db_connection.get_connection()
        cursor = conn.cursor()
        sql = """
            SELECT * FROM backtest_results
            WHERE strategy_id = %s
        """
        values = (strategy_id,)
        cursor.execute(sql, values)
        results = cursor.fetchall()
        cursor.close()
        return results