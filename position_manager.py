from utils import handle_error
from exceptions import OrderError

class PositionManager:
    @handle_error
    def __init__(self, initial_balance, risk_per_trade):
        self.balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.position = None

    @handle_error
    def open_position(self, symbol, side, price, size):
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
    def close_position(self, symbol, price):
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
        return self.position

    @handle_error
    @handle_error
    def get_balance(self):
        return self.balance