import numpy as np

from src.risk.risk_calculator import RiskCalculator


def test_var_and_mdd() -> None:
    calc = RiskCalculator(confidence=0.95)
    returns = np.array([0.01, -0.02, 0.015, -0.03])
    var = calc.value_at_risk(returns)
    assert var > 0

    equity = np.array([100, 90, 95, 80])
    mdd = calc.max_drawdown(equity)
    assert mdd < 0


def test_position_concentration() -> None:
    calc = RiskCalculator()
    concentration = calc.position_concentration([50, 30, 20])
    assert concentration == 0.5
