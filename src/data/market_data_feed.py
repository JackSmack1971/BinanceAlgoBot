from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from typing import Any

import pandas as pd
from binance.client import Client

from ..cache import IntelligentRedisCache
from config import get_config
from performance_utils import log_memory_usage
from utils import handle_error
from exceptions import DataRetrievalError, DataError

logger = logging.getLogger(__name__)


class MarketDataFeed:
    """Data feed with Redis caching and checksums."""

    def __init__(
        self,
        client: Client,
        symbol: str | None = None,
        interval: str | None = None,
        cache: IntelligentRedisCache | None = None,
    ) -> None:
        from config import BINANCE_CONSTANTS

        self.client = client
        self.symbol = symbol or get_config("default_symbol")
        self.interval = interval or BINANCE_CONSTANTS["KLINE_INTERVAL_15MINUTE"]
        self.max_rows = int(os.getenv("DATAFRAME_MAX_ROWS", "1000"))
        self.cache = cache or IntelligentRedisCache()
        logger.info("Initialized MarketDataFeed %s %s", self.symbol, self.interval)

    async def _fetch_klines(self) -> list[Any]:
        try:
            return await asyncio.to_thread(
                self.client.get_klines, symbol=self.symbol, interval=self.interval
            )
        except Exception as exc:
            logger.warning("Data retrieval failed: %s", exc)
            raise DataRetrievalError("Failed to fetch klines") from exc

    def _prepare_dataframe(self, klines: list[Any]) -> pd.DataFrame:
        data = pd.DataFrame(
            klines,
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_asset_volume",
                "trades",
                "taker_buy_base",
                "taker_buy_quote",
                "ignored",
            ],
        )
        data = data.astype(float)
        data.fillna(method="ffill", inplace=True)
        return data.head(self.max_rows)

    def _checksum(self, data: list[Any]) -> str:
        return hashlib.sha256(str(data).encode()).hexdigest()

    @handle_error
    async def get_data(self) -> pd.DataFrame:
        params = {"symbol": self.symbol, "interval": self.interval}
        cached = await self.cache.get("market", params)
        if cached is not None:
            return cached
        klines = await self._fetch_klines()
        df = self._prepare_dataframe(klines)
        await self.cache.set("market", params | {"cs": self._checksum(klines)}, df, 300)
        log_memory_usage("MarketDataFeed: ")
        return df
