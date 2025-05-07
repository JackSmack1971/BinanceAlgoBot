from database.database_connection import DatabaseConnection

class MarketDataRepository:
    def __init__(self):
        self.db_connection = DatabaseConnection()

    def insert_market_data(self, symbol, interval, timestamp, open, high, low, close, volume, close_time, quote_asset_volume, trades, taker_buy_base, taker_buy_quote, ignored):
        conn = self.db_connection.get_connection()
        cursor = conn.cursor()
        sql = """
            INSERT INTO market_data (symbol, interval, timestamp, open, high, low, close, volume, close_time, quote_asset_volume, trades, taker_buy_base, taker_buy_quote, ignored)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = (symbol, interval, timestamp, open, high, low, close, volume, close_time, quote_asset_volume, trades, taker_buy_base, taker_buy_quote, ignored)
        cursor.execute(sql, values)
        conn.commit()
        cursor.close()

    def get_market_data(self, symbol, interval, start_time, end_time):
        conn = self.db_connection.get_connection()
        cursor = conn.cursor()
        sql = """
            SELECT * FROM market_data
            WHERE symbol = %s AND interval = %s AND timestamp >= %s AND timestamp <= %s
        """
        values = (symbol, interval, start_time, end_time)
        cursor.execute(sql, values)
        results = cursor.fetchall()
        cursor.close()
        return results