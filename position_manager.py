from utils import handle_error
from exceptions import OrderError
from validation import validate_symbol, validate_quantity, validate_risk

class PositionManager:
    """Manage open positions and account balance."""

    @handle_error
    def __init__(self, initial_balance: float, risk_per_trade: float) -> None:
        """Create a new ``PositionManager``.

        Parameters
        ----------
        initial_balance : float
            Starting account balance.
        risk_per_trade : float
            Fraction of balance to risk per trade.
        """
        self.balance = validate_quantity(initial_balance)
        self.risk_per_trade = validate_risk(risk_per_trade)
        self.position = None

    @handle_error
    def open_position(self, symbol: str, side: str, price: float, size: float) -> None:
        """Open a new position if none exists."""
        symbol = validate_symbol(symbol)
        price = validate_quantity(price)
        size = validate_quantity(size)
        if self.position:
            raise OrderError("Position already open")
        self.position = {
            "symbol": symbol,
            "side": side,
            "entry_price": price,
            "size": size,
        }
        self.balance -= size * price  # Assuming cost basis
        print(f"Opened {side} position for {symbol} at {price} with size {size}")

    @handle_error
    def close_position(self, symbol: str, price: float) -> float:
        """Close the current position and return the profit."""
        symbol = validate_symbol(symbol)
        price = validate_quantity(price)
        if not self.position:
            raise OrderError("No position open")
        if self.position["symbol"] != symbol:
            raise OrderError(f"Trying to close position for {symbol} but current position is for {self.position['symbol']}")

        side = self.position["side"]
        size = self.position["size"]
        entry_price = self.position["entry_price"]

        if side == "buy":
            profit = (price - entry_price) * size
        else:
            profit = (entry_price - price) * size

        self.balance += size * price # Return cost basis
        self.balance += profit
        self.position = None
        print(f"Closed position for {symbol} at {price} with profit {profit}")
        return profit

    @handle_error
    def get_position(self):
        """Return the currently open position or ``None``."""
        return self.position

    @handle_error
    def get_balance(self):
        """Return the current account balance."""
        return self.balance