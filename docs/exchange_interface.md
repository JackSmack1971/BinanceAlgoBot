# ExchangeInterface

The `ExchangeInterface` defines the abstract base class for interacting with an exchange.

## Methods

### `fetch_market_data()`

```python
@abc.abstractmethod
def fetch_market_data(self, symbol):
    """Fetch market data for a given symbol."""
    raise NotImplementedError
```

Fetches market data for a given symbol.

**Args:**

*   `symbol` (`str`): The trading symbol to fetch market data for.

**Raises:**

*   `NotImplementedError`: If the method is not implemented.

### `place_order()`

```python
@abc.abstractmethod
def place_order(self, symbol, side, quantity, order_type, price=None):
    """Place an order."""
    raise NotImplementedError
```

Places an order.

**Args:**

*   `symbol` (`str`): The trading symbol to place the order for.
*   `side` (`str`): The side of the order ("buy" or "sell").
*   `quantity` (`float`): The quantity to trade.
*   `order_type` (`str`): The type of order to place (e.g., "market", "limit").
*   `price` (`float`, optional): The price at which to place the order (for limit orders). Defaults to `None`.

**Raises:**

*   `NotImplementedError`: If the method is not implemented.

### `get_account_balance()`

```python
@abc.abstractmethod
def get_account_balance(self):
    """Retrieve account balance."""
    raise NotImplementedError
```

Retrieves the account balance.

**Raises:**

*   `NotImplementedError`: If the method is not implemented.

### `get_order_status()`

```python
@abc.abstractmethod
def get_order_status(self, order_id):
    """Get the status of an order."""
    raise NotImplementedError
```

Gets the status of an order.

**Args:**

*   `order_id` (`str`): The ID of the order to get the status for.

**Raises:**

*   `NotImplementedError`: If the method is not implemented.