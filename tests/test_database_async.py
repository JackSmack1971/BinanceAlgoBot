import asyncio
from typing import Any

import pytest

from database.database_connection import DatabaseConnection


class DummyConn:
    async def fetch(self, query: str, *params: Any):
        return [params]


class DummyPool:
    def __init__(self) -> None:
        self.connection = DummyConn()

    async def acquire(self):
        return self.connection

    async def release(self, conn):
        pass

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_execute_query(monkeypatch):
    db = DatabaseConnection()

    async def fake_create_pool(*args, **kwargs):
        return DummyPool()

    monkeypatch.setattr(db, "connect", lambda: fake_create_pool())
    db.conn_pool = await fake_create_pool()
    result = await db.execute_query("SELECT", [1])
    assert result == [(1,)]

