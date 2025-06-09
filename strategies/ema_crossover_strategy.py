import logging
from binance.client import Client
from pandas import DataFrame
from ta.trend import EMAIndicator

from data_feed import DataFeed
from indicators import TechnicalIndicators
from utils import handle_error
from decimal import Decimal
from exceptions import DataError, StrategyError
from config import get_config
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class EMACrossoverStrategy(BaseStrategy):
    """EMA crossover strategy."""

    @handle_error
    def __init__(
        self,
        client: Client,
        symbol: str,
        interval: str,
        fast_ema_window: int | None = None,
        slow_ema_window: int | None = None,
        initial_balance: float = 10000,
        risk_per_trade: float = 0.01,
    ) -> None:
        super().__init__(client, symbol, interval, initial_balance, risk_per_trade)
        self.fast_ema_window = fast_ema_window or 9
        self.slow_ema_window = slow_ema_window or 21
        self.indicators = TechnicalIndicators()
        self.data_feed = DataFeed(client=client, symbol=symbol, interval=interval)

    @handle_error
    def calculate_indicators(self) -> DataFrame | None:
        try:
            data = self.data_feed.get_data()
            if data is None or data.empty:
                return None
            fast_ema = EMAIndicator(close=data["close"], window=self.fast_ema_window, fillna=True)
            slow_ema = EMAIndicator(close=data["close"], window=self.slow_ema_window, fillna=True)
            data["fast_ema"] = fast_ema.ema_indicator()
            data["slow_ema"] = slow_ema.ema_indicator()
            return data
        except Exception as exc:  # pragma: no cover
            logger.error("Error calculating EMA indicators: %s", exc, exc_info=True)
            raise DataError(f"Failed to calculate EMA indicators: {exc}") from exc

    @handle_error
    def run(self) -> DataFrame | None:
        try:
            data = self.calculate_indicators()
            if data is None or data.empty:
                logger.warning("No data available to generate trading signals for %s", self.symbol)
                return None
            data["signal"] = 0.0
            for i in range(1, len(data)):
                if data["fast_ema"].iloc[i] > data["slow_ema"].iloc[i] and data["fast_ema"].iloc[i - 1] <= data["slow_ema"].iloc[i - 1]:
                    data.loc[data.index[i], "signal"] = 1.0
                    if await self.position_manager.get_position(self.symbol) is None:
                        price = data["close"].iloc[i]
                        size = self.position_manager.calculate_position_size(price)
                        await self.open_position("buy", price, size)
                elif data["fast_ema"].iloc[i] < data["slow_ema"].iloc[i] and data["fast_ema"].iloc[i - 1] >= data["slow_ema"].iloc[i - 1]:
                    data.loc[data.index[i], "signal"] = -1.0
                    if await self.position_manager.get_position(self.symbol) is None:
                        price = data["close"].iloc[i]
                        size = self.position_manager.calculate_position_size(price)
                        await self.open_position("sell", price, size)
            for i in range(1, len(data)):
                if data["fast_ema"].iloc[i] < data["slow_ema"].iloc[i] and data["fast_ema"].iloc[i - 1] >= data["slow_ema"].iloc[i - 1]:
                    if await self.position_manager.get_position(self.symbol) is not None:
                        price = data["close"].iloc[i]
                        await self.close_position(price)
                elif data["fast_ema"].iloc[i] > data["slow_ema"].iloc[i] and data["fast_ema"].iloc[i - 1] <= data["slow_ema"].iloc[i - 1]:
                    if await self.position_manager.get_position(self.symbol) is not None:
                        price = data["close"].iloc[i]
                        await self.close_position(price)
            data["position"] = 0
            if data["signal"].iloc[0] != 0:
                data.loc[data.index[0], "position"] = data["signal"].iloc[0]
            for i in range(1, len(data)):
                if data["signal"].iloc[i] != 0:
                    data.loc[data.index[i], "position"] = data["signal"].iloc[i]
                else:
                    data.loc[data.index[i], "position"] = data["position"].iloc[i - 1]
            logger.info("Generated trading signals for %s on %s timeframe", self.symbol, self.interval)
            return data
        except Exception as exc:  # pragma: no cover
            logger.error("Error generating trading signals: %s", exc, exc_info=True)
            raise StrategyError(f"Error generating trading signals: {exc}") from exc

    async def open_position(self, side: str, price: float, size: float) -> None:
        await self.position_manager.open_position(
            self.symbol,
            side,
            Decimal(str(price)),
            Decimal(str(size)),
            {},
        )

    async def close_position(self, price: float) -> None:
        await self.position_manager.close_position(
            self.symbol, Decimal(str(price))
        )
