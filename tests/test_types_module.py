import pytest
from pydantic import ValidationError

from src.trading.types import Price, runtime_type_check


@runtime_type_check
def add_prices(a: Price, b: Price) -> Price:
    return Price(a + b)


def test_runtime_type_check_valid() -> None:
    assert add_prices(Price(1.0), Price(2.0)) == Price(3.0)


def test_runtime_type_check_invalid() -> None:
    with pytest.raises(ValidationError):
        add_prices("oops", Price(2.0))  # type: ignore[arg-type]
