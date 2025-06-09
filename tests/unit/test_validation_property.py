from hypothesis import given, strategies as st
from validation import validate_quantity

@given(st.floats(min_value=0.0001, max_value=1e6, allow_nan=False, allow_infinity=False))
def test_validate_quantity_positive(value: float) -> None:
    assert validate_quantity(value) == float(value)

def test_validate_quantity_invalid() -> None:
    try:
        validate_quantity(0)
    except ValueError:
        assert True
    else:
        assert False
