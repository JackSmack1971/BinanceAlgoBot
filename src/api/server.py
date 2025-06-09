from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import asyncio
import os
from typing import Dict

from src.validation.input_validator import TradingSymbolValidator, verify_signature
from src.database.repositories.secure_market_data_repository import SecureMarketDataRepository

app = FastAPI()

RATE_LIMIT: Dict[str, int] = {}
MAX_REQUESTS = 5
WINDOW_SECONDS = 60


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        client = request.client.host
        RATE_LIMIT.setdefault(client, 0)
        RATE_LIMIT[client] += 1
        await asyncio.sleep(0)  # yield control
        if RATE_LIMIT[client] > MAX_REQUESTS:
            return JSONResponse(status_code=429, content={"detail": "rate limit"})
        response = await call_next(request)
        return response


app.add_middleware(RateLimitMiddleware)


async def signature_dependency(request: Request) -> None:
    secret = os.getenv("API_SECRET", "")
    sig = request.headers.get("X-Signature", "")
    if not verify_signature(request.url.path, sig, secret):
        raise HTTPException(status_code=401, detail="Invalid signature")


@app.get("/market_data", dependencies=[Depends(signature_dependency)])
async def market_data(symbol: str):
    try:
        validated = TradingSymbolValidator(symbol=symbol).symbol
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    async with SecureMarketDataRepository() as repo:
        data = await repo.fetch_by_symbol(validated)
    return {"data": data}
