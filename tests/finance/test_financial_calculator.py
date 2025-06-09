import pytest
from decimal import Decimal

from src.finance.financial_calculator import FinancialCalculator


def test_order_value_and_commission():
    calc = FinancialCalculator()
    value = calc.calculate_order_value('10000', '0.005')
    commission = calc.calculate_commission(value)
    assert value == Decimal('50.00')
    assert commission == Decimal('0.05')


def test_profit_loss_precision():
    calc = FinancialCalculator()
    pnl = calc.calculate_profit_loss(Decimal('10000'), Decimal('10200'), Decimal('1'), 'BUY')
    assert pnl['absolute_pnl'] == Decimal('200.00')
    assert pnl['percentage_pnl'] == Decimal('2.000000')
    assert calc.validate_financial_precision(pnl['percentage_pnl'], 6)

