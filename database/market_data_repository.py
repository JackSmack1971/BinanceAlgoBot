from database.database_connection import DatabaseConnection

class MarketDataRepository:
    def __init__(self):
        self.db_connection = DatabaseConnection()

    async def __aenter__(self):
        await self.db_connection.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.db_connection.disconnect()

    async def insert_market_data(self, market_data_list):
        sql = """
            INSERT INTO market_data (symbol, interval, timestamp, open, high, low, close, volume, close_time, quote_asset_volume, trades, taker_buy_base, taker_buy_quote, ignored)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
        """
        values = [(data['symbol'], data['interval'], data['timestamp'], data['open'], data['high'], data['low'], data['close'], data['volume'], data['close_time'], data['quote_asset_volume'], data['trades'], data['taker_buy_base'], data['taker_buy_quote'], data['ignored']) for data in market_data_list]
        await self.db_connection.execute_query(sql, values)

    async def get_market_data(self, symbol, interval, start_time, end_time, page_number=1, page_size=100):
        offset = (page_number - 1) * page_size
        sql = """
            SELECT * FROM market_data
            WHERE symbol = $1 AND interval = $2 AND timestamp >= $3 AND timestamp <= $4
            LIMIT $5 OFFSET $6
        """
        values = (symbol, interval, start_time, end_time, page_size, offset)
        return await self.db_connection.execute_query(sql, values)