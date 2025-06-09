from __future__ import annotations

from typing import Any, Iterable

from database.database_connection import DatabaseConnection
from src.validation.input_validator import TradingSymbolValidator
from pydantic import ValidationError


class SecureMarketDataRepository:
    """Repository using parameterized queries and validation."""

    def __init__(self) -> None:
        self.db_connection = DatabaseConnection()

    async def __aenter__(self) -> "SecureMarketDataRepository":
        await self.db_connection.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # pragma: no cover
        await self.db_connection.disconnect()

    async def fetch_by_symbol(self, symbol: str) -> Iterable[dict[str, Any]]:
        try:
            symbol_val = TradingSymbolValidator(symbol=symbol).symbol
        except ValidationError as exc:  # pragma: no cover - validation tested
            raise
        query = "SELECT * FROM market_data WHERE symbol = $1"
        async with self.db_connection as conn:
            return await conn.execute_query(query, (symbol_val,))
