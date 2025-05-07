import abc

class ExchangeInterface(abc.ABC):
    @abc.abstractmethod
    def fetch_market_data(self, symbol):
        """Fetch market data for a given symbol."""
        raise NotImplementedError

    @abc.abstractmethod
    def place_order(self, symbol, side, quantity, order_type, price=None):
        """Place an order."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_account_balance(self):
        """Retrieve account balance."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_order_status(self, order_id):
        """Get the status of an order."""
        raise NotImplementedError