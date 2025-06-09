import asyncio
import logging

import aiohttp
import pytest

from src.api.run import _configure_logging, _init_observability


@pytest.mark.asyncio
async def test_init_observability() -> None:
    server = await _init_observability(port=8082)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8082/healthz") as resp:
                assert resp.status == 200
                assert await resp.text() == "ok"
    finally:
        await server.stop()


def test_configure_logging() -> None:
    _configure_logging()
    logger = logging.getLogger()
    assert logger.handlers
    assert logger.handlers[0].formatter.__class__.__name__ == "JsonFormatter"
