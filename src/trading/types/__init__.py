from __future__ import annotations

from functools import wraps
from typing import Any, Callable, NewType

from pydantic import validate_call

# Domain specific types
Price = NewType('Price', float)
Quantity = NewType('Quantity', float)
Symbol = NewType('Symbol', str)


def runtime_type_check(func: Callable[..., Any]) -> Callable[..., Any]:
    """Validate function arguments at runtime using pydantic."""
    validated = validate_call(func)

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return validated(*args, **kwargs)

    return wrapper

__all__ = [
    "Price",
    "Quantity",
    "Symbol",
    "runtime_type_check",
]
