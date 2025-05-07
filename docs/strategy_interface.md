# Strategy Interface

The `Strategy` interface defines the abstract base class for all trading strategies.

## Methods

### `calculate_indicators()`

```python
@abstractmethod
def calculate_indicators(self):
    """
    Calculate indicators required for the strategy.

    Returns:
        pd.DataFrame: DataFrame with calculated indicators
    """
    pass
```

Calculates the indicators required for the strategy.

**Returns:**

*   `pd.DataFrame`: A Pandas DataFrame containing the calculated indicators.

### `run()`

```python
@abstractmethod
def run(self):
    """
    Generate trading signals based on the strategy.

    Returns:
        pd.DataFrame: DataFrame with trading signals
    """
    pass
```

Generates trading signals based on the strategy.

**Returns:**

*   `pd.DataFrame`: A Pandas DataFrame containing the trading signals.

### `open_position()`

```python
@abstractmethod
def open_position(self, side: str, price: float, size: float):
    """
    Open a position.

    Args:
        side (str): "buy" or "sell"
        price (float): Entry price
        size (float): Position size
    """
    pass
```

Opens a position.

**Args:**

*   `side` (`str`): The side of the position to open ("buy" or "sell").
*   `price` (`float`): The entry price for the position.
*   `size` (`float`): The size of the position.

### `close_position()`

```python
@abstractmethod
def close_position(self, price: float):
    """
    Close the current position.

    Args:
        price (float): Exit price
    """
    pass
```

Closes the current position.

**Args:**

*   `price` (`float`): The exit price for the position.