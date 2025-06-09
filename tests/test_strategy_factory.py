import pandas as pd
import pytest
from unittest.mock import Mock, patch

from strategy_factory import StrategyFactory
from strategies.btc_strategy import BTCStrategy


@pytest.fixture
def mock_client() -> Mock:
    return Mock()


def test_create_unknown_strategy(mock_client):
    factory = StrategyFactory()
    strategy = factory.create_strategy("unknown", mock_client, "BTCUSDT", "15m")
    assert strategy is None


def test_btc_strategy_runs(mock_client, sample_strategy_data):
    StrategyFactory.register_strategy("btc_test", BTCStrategy)
    factory = StrategyFactory()
    with patch.object(BTCStrategy, "calculate_indicators", return_value=sample_strategy_data):
        strategy = factory.create_strategy("btc_test", mock_client, "BTCUSDT", "15m")
        signals = strategy.run()
    assert isinstance(signals, pd.DataFrame)
    assert "signal" in signals.columns
