from typing import Iterable, Dict, Any
from validation import validate_symbol, validate_timeframe, validate_quantity

from database.database_connection import DatabaseConnection


class MarketDataRepository:
    def __init__(self) -> None:
        self.db_connection = DatabaseConnection()

    async def __aenter__(self) -> "MarketDataRepository":
        await self.db_connection.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.db_connection.disconnect()

    async def insert_market_data(self, market_data_list: Iterable[Dict[str, Any]]) -> None:
        sql = """
            INSERT INTO market_data (
                symbol, interval, timestamp, open, high, low, close,
                volume, close_time, quote_asset_volume, trades,
                taker_buy_base, taker_buy_quote, ignored
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7,
                $8, $9, $10, $11, $12, $13, $14
            )
        """
        values = [
            (
                validate_symbol(data["symbol"]),
                validate_timeframe(data["interval"]),
                data["timestamp"],
                validate_quantity(data["open"]),
                validate_quantity(data["high"]),
                validate_quantity(data["low"]),
                validate_quantity(data["close"]),
                validate_quantity(data["volume"]),
                data["close_time"],
                validate_quantity(data["quote_asset_volume"]),
                int(validate_quantity(data["trades"])),
                validate_quantity(data["taker_buy_base"]),
                validate_quantity(data["taker_buy_quote"]),
                validate_quantity(data["ignored"]),
            )
            for data in market_data_list
        ]
        async with self.db_connection as conn:
            await conn.execute_batch(sql, values)

    async def get_market_data(
        self,
        symbol: str,
        interval: str,
        start_time: Any,
        end_time: Any,
        page_number: int = 1,
        page_size: int = 100,
    ):
        symbol = validate_symbol(symbol)
        interval = validate_timeframe(interval)
        offset = (page_number - 1) * page_size
        sql = """
            SELECT * FROM market_data
            WHERE symbol = $1 AND interval = $2 AND timestamp >= $3 AND timestamp <= $4
            LIMIT $5 OFFSET $6
        """
        values = (symbol, interval, start_time, end_time, page_size, offset)
        async with self.db_connection as conn:
            return await conn.execute_query(sql, values)

