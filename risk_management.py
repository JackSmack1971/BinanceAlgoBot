# risk_management.py
import logging
from binance.client import Client
from typing import Optional, Dict, Any
from strategy_factory import Strategy  # Fixed import to use the abstract Strategy class
from execution_engine import ExecutionEngine 

logger = logging.getLogger(__name__) 

class RiskManagement:
    """
    Class for managing trading risk based on technical indicators and account constraints.
    """
    def __init__(self, client: Client, strategy: Strategy, engine: ExecutionEngine, max_risk: float):
        """
        Initialize the risk management system.
        
        Args:
            client (Client): Binance API client
            strategy (Strategy): Trading strategy
            engine (ExecutionEngine): Execution engine for trades
            max_risk (float): Maximum risk per trade as a decimal (e.g., 0.01 for 1%)
        """
        self.client = client
        self.strategy = strategy
        self.engine = engine
        self.max_risk = max_risk 

    def manage_risk(self) -> Optional[Dict[str, Any]]:
        """
        Manage risk based on the ATR indicator and account balance.
        
        Returns:
            Optional[Dict[str, Any]]: Risk management details or None if unsuccessful
        """
        try:
            # Calculate indicators to get the current ATR
            data = self.strategy.calculate_indicators()
            
            if data is None or data.empty:
                logger.warning(f"No data available to manage risk for {self.strategy.symbol}")
                return None
                
            # Get the latest ATR value
            atr = data['atr'].iloc[-1]
            
            # Fetch account balance
            account_info = self.client.get_account()
            balances = {asset['asset']: float(asset['free']) for asset in account_info['balances'] 
                       if float(asset['free']) > 0}
            
            # Extract the quote currency from the symbol (e.g., USDT from BTCUSDT)
            quote_currency = self.strategy.symbol[3:] if len(self.strategy.symbol) > 3 else "USDT"
            
            # Get the available balance in the quote currency
            available_balance = balances.get(quote_currency, 0.0)
            
            # Current price
            ticker = self.client.get_symbol_ticker(symbol=self.strategy.symbol)
            current_price = float(ticker['price'])
            
            # Calculate maximum position size based on risk per trade
            risk_amount = available_balance * self.max_risk
            max_position_size = risk_amount / (atr * current_price)
            
            # Apply position sizing constraints
            # 1. Round to appropriate decimal places for the asset
            from config import TRADING_CONFIG
            position_size = round(max_position_size, TRADING_CONFIG.get("quantity_precision", 5))
            
            # 2. Ensure minimum order size is met
            min_order_size = TRADING_CONFIG.get("min_order_size", 0.001)
            position_size = max(position_size, min_order_size)
            
            # Update the quantity in the execution engine
            self.engine.quantity = position_size
            
            logger.info(f"Risk management updated position size to {position_size} for {self.strategy.symbol}")
            
            return {
                'atr': atr,
                'available_balance': available_balance,
                'risk_amount': risk_amount,
                'position_size': position_size
            }
            
        except Exception as e:
            logger.error(f"Error during risk management: {e}", exc_info=True)
            return None


# Only execute this if the script is run directly
if __name__ == "__main__":
    import os
    from config import TRADING_CONFIG, RISK_CONFIG
    from strategy_factory import StrategyFactory
    
    # Initialize client
    client = Client(os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_API_SECRET"))
    
    # Create strategy
    strategy = StrategyFactory.create_strategy(
        TRADING_CONFIG["default_strategy_type"],
        client,
        TRADING_CONFIG["default_symbol"],
        TRADING_CONFIG["default_interval"]
    )
    
    # Initialize components
    engine = ExecutionEngine(client, strategy, TRADING_CONFIG["default_quantity"])
    risk_management = RiskManagement(client, strategy, engine, RISK_CONFIG["max_risk_per_trade"])
    
    # Run risk management
    risk_details = risk_management.manage_risk()
    
    if risk_details:
        print(f"ATR: {risk_details['atr']}")
        print(f"Available Balance: {risk_details['available_balance']} {TRADING_CONFIG['default_symbol'][3:]}")
        print(f"Risk Amount: {risk_details['risk_amount']}")
        print(f"Position Size: {risk_details['position_size']}")
