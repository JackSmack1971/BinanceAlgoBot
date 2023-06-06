from binance.client import Client
from btcstrategy import BTCStrategy 

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
            print(f"An error occurred: {e}")
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
                        print(f"An error occurred: {e}")
                # If the current signal is -1.0, execute a sell order
                elif signals['signal'].iloc[i] == -1.0:
                    try:
                        self.client.order_market_sell(
                            symbol=self.strategy.symbol,
                            quantity=self.quantity
                        )
                    except Exception as e:
                        print(f"An error occurred: {e}")

if __name__ == "__main__":
    client = Client("YOUR_API_KEY", "YOUR_API_SECRET")
    strategy = BTCStrategy(client, "BTCUSDT", Client.KLINE_INTERVAL_15MINUTE)
    engine = ExecutionEngine(client, strategy, 0.001)
    engine.execute_trades()
