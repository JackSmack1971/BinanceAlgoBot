from database.database_connection import DatabaseConnection

class IndicatorRepository:
    def __init__(self):
        self.db_connection = DatabaseConnection()

    def insert_indicator(self, market_data_id, ema, rsi, atr, vwap):
        conn = self.db_connection.get_connection()
        cursor = conn.cursor()
        sql = """
            INSERT INTO indicators (market_data_id, ema, rsi, atr, vwap)
            VALUES (%s, %s, %s, %s, %s)
        """
        values = (market_data_id, ema, rsi, atr, vwap)
        cursor.execute(sql, values)
        conn.commit()
        cursor.close()

    def get_indicators_by_market_data_id(self, market_data_id):
        conn = self.db_connection.get_connection()
        cursor = conn.cursor()
        sql = """
            SELECT * FROM indicators
            WHERE market_data_id = %s
        """
        values = (market_data_id,)
        cursor.execute(sql, values)
        results = cursor.fetchall()
        cursor.close()
        return results