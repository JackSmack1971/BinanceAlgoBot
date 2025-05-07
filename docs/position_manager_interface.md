# PositionManager

The `PositionManager` class is responsible for managing positions, calculating position sizes, and managing risk.

## Methods

### `calculate_position_size()`

```python
def calculate_position_size(self, price):
    """
    Calculate the position size based on the risk per trade and the current price.

    Args:
        price (float): The current price.

    Returns:
        float: The calculated position size.
    """
    return self.risk_per_trade * self.balance / price
```

Calculates the position size based on the risk per trade and the current price.

**Args:**

*   `price` (`float`): The current price.

**Returns:**

*   `float`: The calculated position size.

### `open_position()`

```python
def open_position(self, side: str, price: float, size: float):
    """
    Open a position.

    Args:
        side (str): "buy" or "sell"
        price (float): Entry price
        size (float): Position size
    """
    # Implementation details
    pass
```

Opens a position.

**Args:**

*   `side` (`str`): The side of the position to open ("buy" or "sell").
*   `price` (`float`): The entry price for the position.
*   `size` (`float`): The size of the position.

### `close_position()`

```python
def close_position(self, price: float):
    """
    Close the current position.

    Args:
        price (float): Exit price
    """
    # Implementation details
    pass
```

Closes the current position.

**Args:**

*   `price` (`float`): The exit price for the position.