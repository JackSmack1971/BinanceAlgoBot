from database.database_connection import DatabaseConnection

class StrategyRepository:
    def __init__(self):
        self.db_connection = DatabaseConnection()

    def insert_strategy(self, strategy_name, strategy_type, symbol, interval, initial_balance, risk_per_trade):
        conn = self.db_connection.get_connection()
        cursor = conn.cursor()
        sql = """
            INSERT INTO strategies (strategy_name, strategy_type, symbol, interval, initial_balance, risk_per_trade)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        values = (strategy_name, strategy_type, symbol, interval, initial_balance, risk_per_trade)
        cursor.execute(sql, values)
        conn.commit()
        cursor.close()

    def get_strategy_by_name(self, strategy_name):
        conn = self.db_connection.get_connection()
        cursor = conn.cursor()
        sql = """
            SELECT * FROM strategies
            WHERE strategy_name = %s
        """
        values = (strategy_name,)
        cursor.execute(sql, values)
        results = cursor.fetchall()
        cursor.close()
        return results