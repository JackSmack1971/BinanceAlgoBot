from __future__ import annotations
import asyncio
import logging
import os

import uvicorn
from pythonjsonlogger import jsonlogger

from src.monitoring import start_metrics_server, HealthServer, setup_tracing, setup_sentry

class StartupError(Exception):
    """Raised when the server fails to start."""


async def _init_observability(port: int = 8000) -> HealthServer:
    await start_metrics_server()
    server = HealthServer(port=port)
    await server.start()
    try:
        setup_tracing("trading-bot")
    except Exception as exc:
        logging.getLogger(__name__).warning("Tracing init failed: %s", exc)
    setup_sentry()
    return server


def _configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(jsonlogger.JsonFormatter())  # type: ignore[no-untyped-call]
    root = logging.getLogger()
    root.setLevel(os.getenv("LOG_LEVEL", "INFO"))
    root.handlers = [handler]


async def main() -> None:
    _configure_logging()
    try:
        await _init_observability()
    except Exception as exc:  # pragma: no cover - startup should succeed
        raise StartupError(str(exc)) from exc
    config = uvicorn.Config("src.api.server:app", host="0.0.0.0", port=8000, log_config=None)  # nosec B104
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
