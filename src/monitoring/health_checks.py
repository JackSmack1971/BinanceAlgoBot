from __future__ import annotations

from aiohttp import web
from pydantic import BaseModel, ValidationError, field_validator


class HealthServerError(Exception):
    """Raised when the health server encounters an error."""


class _PortModel(BaseModel):
    port: int

    @field_validator("port")
    @classmethod
    def _validate_port(cls, v: int) -> int:
        if not 1 <= v <= 65535:
            raise ValueError("port out of range")
        return v


class HealthServer:
    def __init__(self, port: int = 8000) -> None:
        try:
            self.config = _PortModel(port=port)
        except ValidationError as exc:
            raise HealthServerError(str(exc))
        self._app = web.Application()
        self._app.router.add_get("/healthz", self._health)
        self._app.router.add_get("/readyz", self._ready)
        self.runner: web.AppRunner | None = None

    async def _health(self, _: web.Request) -> web.Response:
        return web.Response(text="ok")

    async def _ready(self, _: web.Request) -> web.Response:
        return web.Response(text="ready")

    async def start(self) -> None:
        self.runner = web.AppRunner(self._app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "0.0.0.0", self.config.port)
        try:
            await site.start()
        except Exception as exc:  # pragma: no cover - cannot easily trigger
            raise HealthServerError("failed to start") from exc

    async def stop(self) -> None:
        if self.runner:
            await self.runner.cleanup()
