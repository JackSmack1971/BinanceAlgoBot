import logging
from typing import Dict, Type, Optional
from binance.client import Client

from validation import validate_symbol, validate_timeframe, validate_quantity, validate_risk
from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class StrategyFactory:
    """Factory for creating trading strategies."""

    _strategies: Dict[str, Type[BaseStrategy]] = {}

    @classmethod
    def register_strategy(cls, strategy_type: str, strategy_class: Type[BaseStrategy]) -> None:
        cls._strategies[strategy_type] = strategy_class
        logger.info("Registered strategy type: %s", strategy_type)

    @classmethod
    def get_strategy_types(cls) -> list[str]:
        return list(cls._strategies.keys())

    @classmethod
    def create_strategy(
        cls,
        strategy_type: str,
        client: Client,
        symbol: str,
        interval: str,
        initial_balance: float = 10000,
        risk_per_trade: float = 0.01,
        **kwargs,
    ) -> Optional[BaseStrategy]:
        strategy_class = cls._strategies.get(strategy_type)
        if not strategy_class:
            logger.error("Strategy type '%s' not registered.", strategy_type)
            return None
        symbol = validate_symbol(symbol)
        interval = validate_timeframe(interval)
        initial_balance = validate_quantity(initial_balance)
        risk_per_trade = validate_risk(risk_per_trade)
        return strategy_class(client, symbol, interval, initial_balance, risk_per_trade, **kwargs)

from strategies import BTCStrategy, EMACrossoverStrategy, MACDStrategy

StrategyFactory.register_strategy("btc", BTCStrategy)
StrategyFactory.register_strategy("ema_cross", EMACrossoverStrategy)
StrategyFactory.register_strategy("macd", MACDStrategy)
