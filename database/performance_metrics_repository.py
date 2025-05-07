from database.database_connection import DatabaseConnection

class PerformanceMetricsRepository:
    def __init__(self):
        self.db_connection = DatabaseConnection()

    def insert_performance_metrics(self, trade_id, initial_capital, final_capital, total_return, annual_return, max_drawdown, sharpe_ratio, win_rate, avg_profit_pct, risk_reward_ratio, profit_factor):
        conn = self.db_connection.get_connection()
        cursor = conn.cursor()
        sql = """
            INSERT INTO performance_metrics (trade_id, initial_capital, final_capital, total_return, annual_return, max_drawdown, sharpe_ratio, win_rate, avg_profit_pct, risk_reward_ratio, profit_factor)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = (trade_id, initial_capital, final_capital, total_return, annual_return, max_drawdown, sharpe_ratio, win_rate, avg_profit_pct, risk_reward_ratio, profit_factor)
        cursor.execute(sql, values)
        conn.commit()
        cursor.close()

    def get_performance_metrics_by_trade_id(self, trade_id):
        conn = self.db_connection.get_connection()
        cursor = conn.cursor()
        sql = """
            SELECT * FROM performance_metrics
            WHERE trade_id = %s
        """
        values = (trade_id,)
        cursor.execute(sql, values)
        results = cursor.fetchall()
        cursor.close()
        return results