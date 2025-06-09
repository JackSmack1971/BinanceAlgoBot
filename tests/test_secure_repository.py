import pytest

from database.database_connection import DatabaseConnection
from src.database.repositories.secure_market_data_repository import SecureMarketDataRepository


class DummyDB(DatabaseConnection):
    def __init__(self):
        super().__init__()
        self.query = None
        self.params = None

    async def execute_query(self, query, params=None):
        self.query = query
        self.params = params
        return [query, params]


@pytest.mark.asyncio
async def test_fetch_by_symbol():
    repo = SecureMarketDataRepository()
    dummy = DummyDB()
    repo.db_connection = dummy
    result = await repo.fetch_by_symbol("BTCUSDT")
    assert dummy.query == "SELECT * FROM market_data WHERE symbol = $1"
    assert dummy.params == ("BTCUSDT",)
    assert result == [dummy.query, dummy.params]
