# main.py
import os
import logging.config
import argparse
from datetime import datetime
from binance.client import Client
from strategy_factory import StrategyFactory
from trading_orchestrator import TradingOrchestrator
from backtester import Backtester
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
    
    parser.add_argument('--mode', choices=['live', 'backtest', 'compare'], default='backtest',
                        help='Trading mode: live, backtest, or compare (default: backtest)')
    
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
    
    parser.add_argument('--initial-capital', type=float, default=10000.0,
                        help='Initial capital for backtesting (default: 10000.0)')
    
    parser.add_argument('--commission', type=float, default=0.001,
                        help='Commission rate per trade (default: 0.001 = 0.1%%)')
    
    parser.add_argument('--output-dir', type=str, default='backtest_results',
                        help='Directory to save backtest results (default: backtest_results)')
    
    parser.add_argument('--compare-strategies', type=str, nargs='+',
                        help='List of strategies to compare (space-separated)')
    
    return parser.parse_args()

def run_backtest(client, args):
    """
    Run backtest with the specified parameters.
    
    Args:
        client (Client): Binance API client
        args (Namespace): Command line arguments
        
    Returns:
        Dict[str, Any]: Backtest results
    """
    logger = logging.getLogger(__name__)
    
    # Create strategy using the factory
    strategy = StrategyFactory.create_strategy(
        args.strategy, 
        client, 
        args.symbol, 
        args.interval
    )
    
    if strategy is None:
        logger.error(f"Failed to create strategy: {args.strategy}")
        return None
    
    # Create backtester
    backtester = Backtester(client, strategy)
    
    # Run backtest
    print(f"\nRunning backtest for {args.symbol} on {args.interval} timeframe using {args.strategy} strategy")
    print(f"Period: {args.backtest_start} to {args.backtest_end}")
    print(f"Initial capital: ${args.initial_capital}")
    print(f"Commission rate: {args.commission * 100}%")
    
    results = backtester.run(
        start_date=args.backtest_start,
        end_date=args.backtest_end,
        initial_capital=args.initial_capital,
        commission=args.commission
    )
    
    if not results['success']:
        logger.error(f"Backtest failed: {results.get('message', 'Unknown error')}")
        return None
    
    # Add strategy information to results
    results['strategy_name'] = args.strategy
    results['symbol'] = args.symbol
    results['interval'] = args.interval
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_filename = f"{args.symbol}_{args.interval}_{args.strategy}_{timestamp}"
    
    # Save results to files
    report_file = os.path.join(args.output_dir, f"{base_filename}_report.md")
    plot_file = os.path.join(args.output_dir, f"{base_filename}_plot.png")
    
    # Generate and save report
    backtester.generate_report(results, report_file)
    
    # Plot and save results
    backtester.plot_results(results, plot_file)
    
    # Print summary
    metrics = results['metrics']
    print("\nBacktest Results Summary:")
    print(f"Total Return: {metrics['total_return_pct']:.2f}%")
    print(f"Annual Return: {metrics['annual_return_pct']:.2f}%")
    print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate_pct']:.2f}%")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    
    print(f"\nDetailed report saved to: {report_file}")
    print(f"Plot saved to: {plot_file}")
    
    return results

def compare_strategies(client, args):
    """
    Compare multiple strategies.
    
    Args:
        client (Client): Binance API client
        args (Namespace): Command line arguments
    """
    logger = logging.getLogger(__name__)
    
    # Get list of strategies to compare
    strategies = args.compare_strategies if args.compare_strategies else list(STRATEGY_TYPES.keys())
    
    print(f"\nComparing strategies: {', '.join(strategies)}")
    print(f"Symbol: {args.symbol}")
    print(f"Interval: {args.interval}")
    print(f"Period: {args.backtest_start} to {args.backtest_end}")
    
    # List to store backtest results
    backtest_results = []
    
    # Run backtest for each strategy
    for strategy_type in strategies:
        if strategy_type not in STRATEGY_TYPES:
            logger.warning(f"Unknown strategy type: {strategy_type}")
            continue
        
        # Create strategy using the factory
        strategy = StrategyFactory.create_strategy(
            strategy_type, 
            client, 
            args.symbol, 
            args.interval
        )
        
        if strategy is None:
            logger.error(f"Failed to create strategy: {strategy_type}")
            continue
        
        # Create backtester
        backtester = Backtester(client, strategy)
        
        # Run backtest
        print(f"\nRunning backtest for {strategy_type} strategy...")
        
        results = backtester.run(
            start_date=args.backtest_start,
            end_date=args.backtest_end,
            initial_capital=args.initial_capital,
            commission=args.commission
        )
        
        if not results['success']:
            logger.error(f"Backtest failed for {strategy_type}: {results.get('message', 'Unknown error')}")
            continue
        
        # Add strategy information to results
        results['strategy_name'] = strategy_type
        results['symbol'] = args.symbol
        results['interval'] = args.interval
        
        # Add to list of results
        backtest_results.append(results)
        
        # Print brief summary
        metrics = results['metrics']
        print(f"  - Total Return: {metrics['total_return_pct']:.2f}%")
        print(f"  - Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  - Win Rate: {metrics['win_rate_pct']:.2f}%")
    
    if not backtest_results:
        logger.error("No successful backtest results to compare")
        return
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_filename = f"comparison_{args.symbol}_{args.interval}_{timestamp}"
    
    # Save comparison report
    comparison_file = os.path.join(args.output_dir, f"{base_filename}_comparison.md")
    Backtester.compare_strategies(backtest_results, comparison_file)
    
    print(f"\nStrategy comparison report saved to: {comparison_file}")

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
    
    # Initialize the Binance client
    from config import API_CONFIG
    if API_CONFIG["use_testnet"]:
        client = Client(api_key, api_secret, testnet=True)
        logger.info("Using Binance testnet")
    else:
        client = Client(api_key, api_secret)
        logger.info("Using Binance production API")
    
    # Display available strategies
    print("\nAvailable Strategies:")
    for strategy_type, description in STRATEGY_TYPES.items():
        print(f"  - {strategy_type}: {description}")
    
    # Handle different modes
    if args.mode == 'live':
        # Initialize the trading orchestrator
        orchestrator = TradingOrchestrator(api_key, api_secret, [args.symbol], [args.interval])
        
        if not orchestrator.initialize():
            logger.error("Failed to initialize trading orchestrator")
            return 1
        
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
        
        # Start live trading
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
    
    elif args.mode == 'backtest':
        # Run backtest
        run_backtest(client, args)
    
    elif args.mode == 'compare':
        # Compare strategies
        compare_strategies(client, args)
    
    return 0

if __name__ == "__main__":
    exit(main())
