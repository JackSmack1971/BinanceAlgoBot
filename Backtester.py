import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Type, Tuple
from datetime import datetime
from binance.client import Client

from strategy_factory import Strategy
from config import get_config

logger = logging.getLogger(__name__)

class Backtester:
    """
    Class for backtesting trading strategies using historical data.
    """

    def __init__(self, client: Client, strategy: Strategy, initial_capital: float = 10000.0, commission: float = 0.001):
        """
        Initialize the backtester.

        Args:
            client (Client): Binance API client
            strategy (Strategy): Trading strategy to backtest
        """
        self.client = client
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission

        self.backtest_executor = BacktestExecutor(client, strategy)
        self.performance_analyzer = PerformanceAnalyzer(initial_capital, commission)
        self.report_generator = ReportGenerator()
        self.visualization_service = VisualizationService()

        logger.info(f"Initialized Backtester for {strategy.__class__.__name__} on {strategy.symbol}/{strategy.interval}")

    def run(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Run the backtest for the specified period.

        Args:
            start_date (str): Start date for backtesting (format: 'YYYY-MM-DD')
            end_date (str): End date for backtesting (format: 'YYYY-MM-DD')

        Returns:
            Dict[str, Any]: Backtesting results
        """
        try:
            logger.info(f"Running backtest from {start_date} to {end_date}")

            # Execute the backtest
            signals = self.backtest_executor.run(start_date, end_date)

            if signals.empty:
                logger.error(f"No signals generated for {self.strategy.symbol}")
                return {
                    'success': False,
                    'message': f"No signals generated for {self.strategy.symbol}"
                }

            # Analyze performance
            performance_results = self.performance_analyzer.calculate_performance(signals)

            if not performance_results:
                logger.error(f"Error during performance analysis for {self.strategy.symbol}")
                return {
                    'success': False,
                    'message': f"Error during performance analysis for {self.strategy.symbol}"
                }

            performance_results['success'] = True
            logger.info(f"Backtest completed for {self.strategy.symbol} from {start_date} to {end_date}")
            return performance_results

        except Exception as e:
            logger.error(f"Error during backtesting: {e}", exc_info=True)
            return {
                'success': False,
                'message': f"Error during backtesting: {str(e)}"
            }

    def plot_results(self, results: Dict[str, Any], filename: str = None) -> bool:
        """
        Plot the backtest results.

        Args:
            results (Dict[str, Any]): Backtest results from the run method
            filename (str, optional): Filename to save the plot. If None, the plot will be displayed.

        Returns:
            bool: True if successful, False otherwise
        """
        if not results['success']:
            logger.error(f"Cannot plot results: {results['message']}")
            return False

        return self.visualization_service.plot_results(results, self.strategy.__class__.__name__, self.strategy.symbol, self.strategy.interval, filename)

    def generate_report(self, results: Dict[str, Any], filename: str = None) -> str:
        """
        Generate a detailed report of the backtest results.

        Args:
            results (Dict[str, Any]): Backtest results from the run method
            filename (str, optional): Filename to save the report. If None, the report will be returned as a string.

        Returns:
            str: Report as a string if filename is None, otherwise the path to the saved report
        """
        if not results['success']:
            return f"Cannot generate report: {results['message']}"

        return self.report_generator.generate_report(results, self.strategy.__class__.__name__, self.strategy.symbol, self.strategy.interval, filename)

    @staticmethod
    def compare_strategies(backtests: List[Dict[str, Any]], filename: str = None) -> Optional[str]:
        """
        Compare multiple backtesting results and generate a combined report.

        Args:
            backtests (List[Dict[str, Any]]): List of backtesting results from the run method
            filename (str, optional): Filename to save the report. If None, the report will be returned as a string.

        Returns:
            Optional[str]: Report as a string if filename is None, otherwise the path to the saved report
        """
        try:
            if not all(backtest['success'] for backtest in backtests):
                return "Cannot generate comparison report: One or more backtests failed."

            # Extract metrics from each backtest
            metrics = [backtest['metrics'] for backtest in backtests]

            # Create a Pandas DataFrame from the metrics
            df = pd.DataFrame(metrics)

            # Calculate summary statistics
            summary = df.describe()

            # Generate report header
            report = "# Strategies Comparison Report\n\n"

            # Add summary statistics to the report
            report += "## Summary Statistics\n\n"
            report += summary.to_string() + "\n\n"

            # Save or return the report
            if filename:
                with open(filename, 'w') as f:
                    f.write(report)
                logger.info(f"Strategies comparison report saved as {filename}")
                return filename
            else:
                return report

        except Exception as e:
            logger.error(f"Error generating comparison report: {e}", exc_info=True)
            return f"Error generating comparison report: {e}"
