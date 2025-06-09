import logging
import os
from typing import Any

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice

from config import get_config
from utils import handle_error
from cache import cache_result
from performance_utils import log_memory_usage

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Utility class for calculating various indicators."""

    @handle_error
    def __init__(self, ema_window: int | None = None, rsi_window: int | None = None, atr_window: int | None = None, vwap_window: int | None = None) -> None:
        self.ema_window = ema_window or get_config("ema_window")
        self.rsi_window = rsi_window or get_config("rsi_window")
        self.atr_window = atr_window or get_config("atr_window")
        self.vwap_window = vwap_window or get_config("vwap_window")


class CachedIndicators(TechnicalIndicators):
    """Indicator calculations with Redis caching."""

    @handle_error
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.max_rows = int(os.getenv("DATAFRAME_MAX_ROWS", "1000"))

    @cache_result(ttl=300)
    async def calculate_ema(self, data: pd.DataFrame) -> pd.DataFrame:
        ema_indicator = EMAIndicator(close=data["close"], window=self.ema_window, fillna=True)
        data["ema"] = ema_indicator.ema_indicator()
        return data

    @cache_result(ttl=300)
    async def calculate_rsi(self, data: pd.DataFrame) -> pd.DataFrame:
        rsi_indicator = RSIIndicator(close=data["close"], window=self.rsi_window, fillna=True)
        data["rsi"] = rsi_indicator.rsi()
        return data

    @cache_result(ttl=300)
    async def calculate_atr(self, data: pd.DataFrame) -> pd.DataFrame:
        atr_indicator = AverageTrueRange(high=data["high"], low=data["low"], close=data["close"], window=self.atr_window, fillna=True)
        data["atr"] = atr_indicator.average_true_range()
        return data

    @cache_result(ttl=300)
    async def calculate_vwap(self, data: pd.DataFrame) -> pd.DataFrame:
        vwap_indicator = VolumeWeightedAveragePrice(high=data["high"], low=data["low"], close=data["close"], volume=data["volume"], window=self.vwap_window, fillna=True)
        data["vwap"] = vwap_indicator.volume_weighted_average_price()
        return data

    @handle_error
    async def calculate_all(self, data: pd.DataFrame) -> pd.DataFrame:
        if data is None or data.empty:
            return data
        data = data.head(self.max_rows)
        data = await self.calculate_ema(data)
        data = await self.calculate_rsi(data)
        data = await self.calculate_atr(data)
        data = await self.calculate_vwap(data)
        from service.service_locator import ServiceLocator

        service_locator = ServiceLocator()
        indicator_service = service_locator.get("IndicatorService")
        market_data_service = service_locator.get("MarketDataService")
        for _, row in data.iterrows():
            market_data = await market_data_service.get_market_data(symbol="BTCUSDT")
            if market_data:
                market_data_id = market_data[0][0]
                await indicator_service.calculate_indicators([
                    {
                        "market_data_id": market_data_id,
                        "ema": row.get("ema", 0),
                        "rsi": row.get("rsi", 0),
                        "atr": row.get("atr", 0),
                        "vwap": row.get("vwap", 0),
                    }
                ])
        log_memory_usage("Indicators: ")
        return data

