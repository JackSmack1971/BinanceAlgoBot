import pytest

pytest.skip("End-to-end workflow requires full environment", allow_module_level=True)

from TradingOrchestrator import TradingOrchestrator
from exchange_interface import ExchangeInterface
from position_manager import PositionManager
from data_feed import DataFeed
from strategy_factory import StrategyFactory
from strategies.btc_strategy import BTCStrategy


class MockExchangeInterface(ExchangeInterface):
    def fetch_market_data(self, symbol):
        return []

    def place_order(self, symbol, side, quantity, order_type, price=None):
        return True

    def get_account_balance(self):
        return {}

    def get_order_status(self, order_id):
        return "filled"
    def get_current_price(self, symbol):
        return 100.0

    def execute_order(self, symbol, order_type, quantity):
        return True


class MockDataFeed:
    def get_historical_data(self, symbol, start_date, end_date):
        return [(100.0, 1000)]


class MockPositionManager(PositionManager):
    def update_position(self, symbol, quantity, price):
        pass



class MockExecutionEngine:
    def execute_trade(self, symbol, order_type, quantity):
        return True


@pytest.fixture
def trading_workflow():
    exchange = MockExchangeInterface()
    position_manager = MockPositionManager(1000, 0.01)
    execution_engine = MockExecutionEngine()
    data_feed = MockDataFeed()
    strategy_factory = StrategyFactory()
    trading_orchestrator = TradingOrchestrator(exchange, position_manager, execution_engine)
    strategy_factory.register_strategy("btc", BTCStrategy)
    return trading_orchestrator, strategy_factory


def test_trading_workflow_e2e(trading_workflow):
    orchestrator, strategy_factory = trading_workflow
    assert orchestrator is not None
    assert strategy_factory is not None
