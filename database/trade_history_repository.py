from database.database_connection import DatabaseConnection

class TradeHistoryRepository:
    def __init__(self):
        self.db_connection = DatabaseConnection()

    def insert_trade_history(self, strategy_id, entry_time, exit_time, position_type, entry_price, exit_price, profit_pct, duration, commission_fee):
        conn = self.db_connection.get_connection()
        cursor = conn.cursor()
        sql = """
            INSERT INTO trade_history (strategy_id, entry_time, exit_time, position_type, entry_price, exit_price, profit_pct, duration, commission_fee)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = (strategy_id, entry_time, exit_time, position_type, entry_price, exit_price, profit_pct, duration, commission_fee)
        cursor.execute(sql, values)
        conn.commit()
        cursor.close()

    def get_trade_history_by_strategy_id(self, strategy_id):
        conn = self.db_connection.get_connection()
        cursor = conn.cursor()
        sql = """
            SELECT * FROM trade_history
            WHERE strategy_id = %s
        """
        values = (strategy_id,)
        cursor.execute(sql, values)
        results = cursor.fetchall()
        cursor.close()
        return results