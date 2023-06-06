import logging
from binance.client import Client
from btcstrategy import BTCStrategy 

logger = logging.getLogger(__name__) 

class TradeExecutionError(Exception):
    """Exception raised when there is an error executing a trade."""
    pass 

class ExecutionEngine:
    def __init__(self, client: Client, strategy: BTCStrategy, quantity: float):
        self.client = client
        self.strategy = strategy
        self.quantity = quantity 

    def execute_trades(self):
        """Execute trades based on signals from the strategy"""
        try:
            signals = self.strategy.run() 
        except Exception as e:
            logger.error(f"An error occurred: {e}", exc_info=True)
            return 

        if signals is None or signals.empty:
            logger.warning("No signals to execute trades")
            return 

        for i in range(1, len(signals)):
            # Check if the current signal is different from the previous one
            if signals['signal'].iloc[i] != signals['signal'].iloc[i-1]:
                # If the current signal is 1.0, execute a buy order
                if signals['signal'].iloc[i] == 1.0:
                    try:
                        self.client.order_market_buy(
                            symbol=self.strategy.symbol,
                            quantity=self.quantity
                        )
                    except Exception as e:
                        logger.error(f"An error occurred: {e}", exc_info=True)
                        raise TradeExecutionError(f"Could not execute buy order for symbol {self.strategy.symbol} and quantity {self.quantity}") from e
                # If the current signal is -1.0, execute a sell order
                elif signals['signal'].iloc[i] == -1.0:
                    try:
                        self.client.order_market_sell(
                            symbol=self.strategy.symbol,
                            quantity=self.quantity
                        )
                    except Exception as e:
                        logger.error(f"An error occurred: {e}", exc_info=True)
                        raise TradeExecutionError(f"Could not execute sell order for symbol {self.strategy.symbol} and quantity {self.quantity}") from e 

if __name__ == "__main__":
    import os
    client = Client(os.getenv("YOUR_API_KEY"), os.getenv("YOUR_API_SECRET"))
    strategy = BTCStrategy(client, "BTCUSDT", Client.KLINE_INTERVAL_15MINUTE)
    engine = ExecutionEngine(client, strategy, 0.001)
    try:
        engine.execute_trades()
    except TradeExecutionError as e:
        print(f"An error occurred while executing trades: {e}")
