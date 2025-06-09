import pytest

from database.performance_metrics_repository import PerformanceMetricsRepository


class DummyDB:
    def __init__(self):
        self.queries = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def execute_query(self, sql, params=None):
        self.queries.append((sql, params))
        return [{"trade_id": params[0]}]


@pytest.mark.asyncio
async def test_insert_and_get_metrics():
    repo = PerformanceMetricsRepository()
    repo.db_connection = DummyDB()

    await repo.insert_performance_metrics(1, 100, 110, 0.1, 0.1, 0.05, 1.0, 0.5, 5, 2, 4)
    assert repo.db_connection.queries

    data = await repo.get_performance_metrics_by_trade_id(1)
    assert data == [{"trade_id": 1}]
