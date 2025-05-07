from database.database_connection import DatabaseConnection

class SignalRepository:
    def __init__(self):
        self.db_connection = DatabaseConnection()

    def insert_signal(self, market_data_id, strategy_id, signal):
        conn = self.db_connection.get_connection()
        cursor = conn.cursor()
        sql = """
            INSERT INTO signals (market_data_id, strategy_id, signal)
            VALUES (%s, %s, %s)
        """
        values = (market_data_id, strategy_id, signal)
        cursor.execute(sql, values)
        conn.commit()
        cursor.close()

    def get_signals_by_strategy_id(self, strategy_id):
        conn = self.db_connection.get_connection()
        cursor = conn.cursor()
        sql = """
            SELECT * FROM signals
            WHERE strategy_id = %s
        """
        values = (strategy_id,)
        cursor.execute(sql, values)
        results = cursor.fetchall()
        cursor.close()
        return results