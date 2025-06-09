import pytest

from src.risk.risk_models import RiskParameters, PositionRisk
from src.trading.types import Price, Quantity, Symbol


def test_risk_parameters_validation() -> None:
    params = RiskParameters(max_risk=0.1, max_position_size=Quantity(1.0), stop_loss=Price(100.0))
    assert params.max_risk == 0.1
    with pytest.raises(ValueError):
        RiskParameters(max_risk=1.5, max_position_size=Quantity(1.0), stop_loss=Price(100.0))


def test_position_risk_loss() -> None:
    risk = PositionRisk(symbol=Symbol("BTCUSDT"), quantity=Quantity(2.0), entry_price=Price(50.0), stop_loss=Price(40.0))
    assert risk.potential_loss() == Price(20.0)
