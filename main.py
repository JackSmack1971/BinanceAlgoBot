import os
import logging.config
import argparse
from binance.client import Client
from strategy_factory import StrategyFactory
from trading_orchestrator import TradingOrchestrator
from config import LOGGING_CONFIG, STRATEGY_TYPES

def configure_logging():
    """Configure logging based on the configuration."""
    logging.config.dictConfig({
        'version': 1,
        'formatters': {
            'standard': {
                'format': LOGGING_CONFIG['format']
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': LOGGING_CONFIG['level'],
                'formatter': 'standard',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.FileHandler',
                'level': LOGGING_CONFIG['level'],
                'formatter': 'standard',
                'filename': LOGGING_CONFIG['log_file'],
                'mode': 'a',
            }
        },
        'loggers': {
            '': {  # root logger
                'handlers': ['console', 'file'] if LOGGING_CONFIG['log_to_file'] else ['console'],
                'level': LOGGING_CONFIG['level'],
                'propagate': True
            }
        }
    })

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Binance Algorithmic Trading Bot')
    
    parser.add_argument('--mode', choices=['live', 'backtest'], default='backtest',
                        help='Trading mode: live or backtest (default: backtest)')
    
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                        help='Trading symbol (default: BTCUSDT)')
    
    parser.add_argument('--interval', type=str, default='15m',
                        help='Timeframe interval (default: 15m)')
    
    parser.add_argument('--strategy', type=str, choices=list(STRATEGY_TYPES.keys()), default='btc',
                        help=f'Strategy type (default: btc). Available types: {", ".join(STRATEGY_TYPES.keys())}')
    
    parser.add_argument('--backtest-start', type=str, default='2023-01-01',
                        help='Start date for backtesting (format: YYYY-MM-DD) (default: 2023-01-01)')
    
    parser.add_argument('--backtest-end', type=str, default='2023-12-31',
                        help='End date for backtesting (format: YYYY-MM-DD) (default: 2023-12-31)')
    
    return parser.parse_args()

def main():
    """Main entry point for the application."""
    # Configure logging
    configure_logging()
    logger = logging.getLogger(__name__)
    
    # Parse command line arguments
    args = parse_args()
    
    # Get API credentials from environment variables
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    
    if not api_key or not api_secret:
        logger.error("API credentials not found in environment variables")
        print("Please set BINANCE_API_KEY and BINANCE_API_SECRET environment variables.")
        return 1
    
    # Initialize the trading orchestrator
    orchestrator = TradingOrchestrator(api_key, api_secret, [args.symbol], [args.interval])
    
    if not orchestrator.initialize():
        logger.error("Failed to initialize trading orchestrator")
        return 1
    
    # Display available strategies
    print("\nAvailable Strategies:")
    for strategy_type, description in STRATEGY_TYPES.items():
        print(f"  - {strategy_type}: {description}")
    
    # If the selected strategy is not the default, change it
    if args.strategy != 'btc':
        logger.info(f"Changing strategy to {args.strategy}")
        success = orchestrator.change_strategy(args.symbol, args.interval, args.strategy)
        if not success:
            logger.error(f"Failed to change strategy to {args.strategy}")
            return 1
    
    # Print account balance
    balance = orchestrator.get_account_balance()
    if balance is not None:
        print(f"\nAccount balance: {balance} USDT")
    
    if args.mode == 'backtest':
        print(f"\nRunning backtest for {args.symbol} on {args.interval} timeframe using {args.strategy} strategy")
        print(f"Period: {args.backtest_start} to {args.backtest_end}")
        
        results = orchestrator.backtest(args.symbol, args.interval, args.backtest_start, args.backtest_end)
        
        if results:
            print("\nBacktest Results:")
            print(f"Total Return: {results['total_return']:.2%}")
            print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
            print(f"Maximum Drawdown: {results['max_drawdown']:.2%}")
            print(f"Number of Trades: {results['num_trades']}")
            
            # Plot results if matplotlib is available
            try:
                import matplotlib.pyplot as plt
                import numpy as np
                
                # Calculate cumulative returns
                cum_returns = (1 + results['data']['returns']).cumprod() - 1
                
                # Create figure and axis
                fig, ax1 = plt.subplots(figsize=(12, 8))
                
                # Plot price and indicators
                ax1.set_title(f"{args.symbol} {args.interval} - {args.strategy} Strategy Backtest")
                ax1.set_xlabel('Date')
                ax1.set_ylabel('Price', color='black')
                ax1.plot(results['data'].index, results['data']['close'], color='black', alpha=0.5, label='Price')
                
                # Plot buy and sell signals
                buy_signals = results['data'][results['data']['signal'] == 1.0]
                sell_signals = results['data'][results['data']['signal'] == -1.0]
                
                ax1.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', label='Buy', alpha=1)
                ax1.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', label='Sell', alpha=1)
                
                # Plot strategy indicators based on the strategy type
                if args.strategy == 'btc':
                    ax1.plot(results['data'].index, results['data']['ema'], color='blue', label='EMA')
                    ax1.plot(results['data'].index, results['data']['vwap'], color='purple', label='VWAP')
                elif args.strategy == 'ema_cross':
                    ax1.plot(results['data'].index, results['data']['fast_ema'], color='blue', label='Fast EMA')
                    ax1.plot(results['data'].index, results['data']['slow_ema'], color='orange', label='Slow EMA')
                elif args.strategy == 'macd':
                    ax1.plot(results['data'].index, results['data']['close'], color='black', alpha=0.5, label='Price')
                
                ax1.legend(loc='upper left')
                ax1.grid(True, alpha=0.3)
                
                # Create a secondary y-axis for returns
                ax2 = ax1.twinx()
                ax2.set_ylabel('Cumulative Returns', color='green')
                ax2.plot(results['data'].index, cum_returns, color='green', linestyle='--', label='Returns')
                ax2.tick_params(axis='y', labelcolor='green')
                
                # If MACD strategy, add a third y-axis for MACD
                if args.strategy == 'macd':
                    ax3 = ax1.twinx()
                    ax3.spines['right'].set_position(('outward', 60))
                    ax3.set_ylabel('MACD', color='blue')
                    ax3.plot(results['data'].index, results['data']['macd'], color='blue', label='MACD')
                    ax3.plot(results['data'].index, results['data']['macd_signal'], color='red', label='Signal')
                    ax3.bar(results['data'].index, results['data']['macd_diff'], color='gray', alpha=0.3, label='Histogram')
                    ax3.tick_params(axis='y', labelcolor='blue')
                    ax3.legend(loc='lower right')
                
                plt.tight_layout()
                plt.savefig(f"{args.symbol}_{args.interval}_{args.strategy}_backtest.png")
                print(f"\nBacktest chart saved as {args.symbol}_{args.interval}_{args.strategy}_backtest.png")
                
            except ImportError:
                print("\nMatplotlib not installed. Install it to generate backtest charts.")
                
    else:  # Live trading
        print(f"\nStarting live trading for {args.symbol} on {args.interval} timeframe using {args.strategy} strategy")
        print("Press Ctrl+C to stop")
        
        try:
            orchestrator.start_trading()
        except KeyboardInterrupt:
            print("\nTrading stopped by user")
            orchestrator.stop_trading()
        except Exception as e:
            logger.error(f"An error occurred: {e}", exc_info=True)
            orchestrator.stop_trading()
    
    return 0

if __name__ == "__main__":
    exit(main())
