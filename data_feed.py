import logging
import os
import asyncio
import pandas as pd
from binance.client import Client
from config import get_config
from exceptions import BaseTradingException, DataError, DataRetrievalError
from src.risk import (
    AdvancedCircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    RateLimitError,
)

logger = logging.getLogger(__name__)

from utils import handle_error
from cache import cache_result
from performance_utils import log_memory_usage


class DataFeed:
    @handle_error
    def __init__(
        self,
        client: Client,
        symbol: str | None = None,
        interval: str | None = None,
        breaker: AdvancedCircuitBreaker | None = None,
    ):
        """
        Initialize the data feed.

        Args:
            client (Client): Binance API client
            symbol (str, optional): Trading symbol (e.g., 'BTCUSDT').
                                   Defaults to config value if None.
            interval (str, optional): Timeframe interval (e.g., '15m').
                                     Defaults to config value if None.
        """
        from config import BINANCE_CONSTANTS

        self.client = client
        self.symbol = symbol if symbol else get_config("default_symbol")
        self.interval = (
            interval if interval else BINANCE_CONSTANTS["KLINE_INTERVAL_15MINUTE"]
        )
        self.breaker = breaker or AdvancedCircuitBreaker(
            "data_feed",
            CircuitBreakerConfig(max_requests_per_minute=120, request_timeout=5.0),
        )
        self.max_rows = int(os.getenv("DATAFRAME_MAX_ROWS", "1000"))
        logger.info(
            "Initialized DataFeed with symbol=%s, interval=%s",
            self.symbol,
            self.interval,
        )

    async def _fetch_klines(self) -> list:
        """Fetch kline data from Binance with circuit breaker protection."""

        async def _call() -> list:
            return await asyncio.to_thread(
                self.client.get_klines,
                symbol=self.symbol,
                interval=self.interval,
            )

        logger.info(
            "Retrieving data for %s on %s timeframe", self.symbol, self.interval
        )
        try:
            return await self.breaker.call(_call)
        except RateLimitError as exc:
            logger.warning("Rate limit hit: %s", exc)
            raise DataRetrievalError("Rate limit exceeded") from exc
        except CircuitBreakerError as exc:
            logger.error("Circuit open: %s", exc)
            raise DataRetrievalError("Service unavailable") from exc
        except Exception as exc:
            logger.warning("Data retrieval failed: %s", exc)
            raise DataRetrievalError(
                f"Could not retrieve data for symbol {self.symbol} and interval {self.interval}"
            ) from exc

    def _prepare_dataframe(self, klines: list) -> pd.DataFrame:
        """Convert raw kline data to a cleaned ``pandas`` DataFrame."""
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

    async def _store_data(self, data: pd.DataFrame) -> None:
        """Persist fetched market data to the database."""
        from database.market_data_repository import MarketDataRepository

        market_data_repo = MarketDataRepository()
        market_data_list = [
            {
                "symbol": self.symbol,
                "interval": self.interval,
                "timestamp": row["timestamp"],
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
                "close_time": row["close_time"],
                "quote_asset_volume": row["quote_asset_volume"],
                "trades": row["trades"],
                "taker_buy_base": row["taker_buy_base"],
                "taker_buy_quote": row["taker_buy_quote"],
                "ignored": row["ignored"],
            }
            for _, row in data.iterrows()
        ]
        try:
            await market_data_repo.insert_market_data(market_data_list)
        except DataError as exc:
            logger.error("Database error storing market data: %s", exc)
            raise

    @cache_result(ttl=300)
    async def get_data(self) -> pd.DataFrame:
        """Return cached market data as a DataFrame."""
        klines = await self._fetch_klines()
        data = self._prepare_dataframe(klines)
        logger.debug("Retrieved %s data points", len(data))
        await self._store_data(data)
        log_memory_usage("DataFeed: ")
        return data
