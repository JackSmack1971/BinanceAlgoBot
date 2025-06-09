import logging
from binance.client import Client
from pandas import DataFrame

from data_feed import DataFeed
from indicators import TechnicalIndicators
from config import get_config
from decimal import Decimal
from utils import handle_error
from exceptions import DataError, StrategyError
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class BTCStrategy(BaseStrategy):
    """Bitcoin trading strategy using EMA, RSI, ATR, and VWAP."""

    @handle_error
    def __init__(
        self,
        client: Client,
        symbol: str,
        interval: str,
        ema_window: int | None = None,
        rsi_window: int | None = None,
        atr_window: int | None = None,
        vwap_window: int | None = None,
        initial_balance: float = 10000,
        risk_per_trade: float = 0.01,
    ) -> None:
        super().__init__(client, symbol, interval, initial_balance, risk_per_trade)
        self.ema_window = ema_window or get_config("ema_window")
        self.rsi_window = rsi_window or get_config("rsi_window")
        self.atr_window = atr_window or get_config("atr_window")
        self.vwap_window = vwap_window or get_config("vwap_window")
        self.indicators = TechnicalIndicators(
            ema_window=self.ema_window,
            rsi_window=self.rsi_window,
            atr_window=self.atr_window,
            vwap_window=self.vwap_window,
        )
        self.data_feed = DataFeed(client=client, symbol=symbol, interval=interval)

    @handle_error
    def calculate_indicators(self) -> DataFrame | None:
        try:
            data = self.data_feed.get_data()
            if data is not None:
                data = self.indicators.calculate_all(data)
            return data
        except Exception as exc:  # pragma: no cover - log and raise custom
            logger.error("Error calculating indicators: %s", exc, exc_info=True)
            raise DataError(f"Failed to calculate indicators: {exc}") from exc

    @handle_error
    def run(self) -> DataFrame | None:
        try:
            data = self.calculate_indicators()
            if data is None or data.empty:
                logger.warning("No data available to generate trading signals for %s", self.symbol)
                return None
            data["signal"] = 0.0
            rsi_buy = get_config("rsi_buy_threshold")
            rsi_sell = get_config("rsi_sell_threshold")
            volatility_increase = get_config("volatility_increase_factor")
            volatility_decrease = get_config("volatility_decrease_factor")
            rsi_oversold = get_config("rsi_oversold")
            for i in range(1, len(data)):
                if (
                    data["close"].iloc[i] > data["vwap"].iloc[i]
                    and data["close"].iloc[i - 1] <= data["vwap"].iloc[i - 1]
                    and data["rsi"].iloc[i] > rsi_buy
                ):
                    data.loc[data.index[i], "signal"] = 1.0
                    if await self.position_manager.get_position(self.symbol) is None:
                        price = data["close"].iloc[i]
                        size = self.position_manager.calculate_position_size(price)
                        await self.open_position("buy", price, size)
                elif (
                    data["close"].iloc[i] < data["vwap"].iloc[i]
                    and data["close"].iloc[i - 1] >= data["vwap"].iloc[i - 1]
                    and data["rsi"].iloc[i] < rsi_sell
                ):
                    data.loc[data.index[i], "signal"] = -1.0
                    if await self.position_manager.get_position(self.symbol) is None:
                        price = data["close"].iloc[i]
                        size = self.position_manager.calculate_position_size(price)
                        await self.open_position("sell", price, size)
                elif (
                    data["close"].iloc[i] > data["ema"].iloc[i]
                    and data["atr"].iloc[i] > data["atr"].iloc[i - 1] * volatility_increase
                ):
                    data.loc[data.index[i], "signal"] = -1.0
                    if await self.position_manager.get_position(self.symbol) is not None:
                        price = data["close"].iloc[i]
                        await self.close_position(price)
                elif (
                    data["close"].iloc[i] < data["ema"].iloc[i]
                    and data["atr"].iloc[i] < data["atr"].iloc[i - 1] * volatility_decrease
                    and data["rsi"].iloc[i] > rsi_oversold
                ):
                    data.loc[data.index[i], "signal"] = 1.0
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
