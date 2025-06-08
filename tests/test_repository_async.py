import pytest

from database.indicator_repository import IndicatorRepository


class DummyDB:
    def __init__(self):
        self.called = False

    async def execute_query(self, query, params=None):
        self.called = True
        return [query, params]

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass


@pytest.mark.asyncio
async def test_indicator_repository(monkeypatch):
    repo = IndicatorRepository()
    dummy = DummyDB()
    repo.db_connection = dummy
    await repo.insert_indicator(1, 1.0, 1.0, 1.0, 1.0)
    assert dummy.called

