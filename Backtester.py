import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Type, Tuple
from datetime import datetime
from binance.client import Client

from utils import handle_error
from exceptions import BaseTradingException

from strategy_factory import Strategy
from config import get_config
from backtest_configuration import BacktestConfiguration
from BacktestExecutor import BacktestExecutor

logger = logging.getLogger(__name__)

class Backtester:
    """
    Class for backtesting trading strategies using historical data.
    """

    from utils import handle_error

    @handle_error
    def __init__(self, config: BacktestConfiguration):
        """
        Initialize the backtester.

        Args:
            config (BacktestConfiguration): Backtest configuration object.
        """
        self.config = config
        self.backtest_executor = BacktestExecutor(config)
        self.performance_analyzer = PerformanceAnalyzer(config.initial_capital, config.commission)
        self.report_generator = ReportGenerator()
        self.visualization_service = VisualizationService()

        logger.info(f"Initialized Backtester for {config.strategy.__class__.__name__} on {config.strategy.symbol}/{config.strategy.interval}")

    def run(self) -> Dict[str, Any]:
        """
        Run the backtest for the specified period.

        Returns:
            Dict[str, Any]: Backtesting results
        """
        try:
            logger.info(f"Running backtest from {self.config.start_date} to {self.config.end_date}")

            # Execute the backtest
            signals = self.backtest_executor.run()

            if signals.empty:
                logger.error(f"No signals generated for {self.config.strategy.symbol}")
                return {
                    'success': False,
                    'message': f"No signals generated for {self.config.strategy.symbol}"
                }

            # Analyze performance
            performance_results = self.performance_analyzer.calculate_performance(signals)

            if not performance_results:
                logger.error(f"Error during performance analysis for {self.config.strategy.symbol}")
                return {
                    'success': False,
                    'message': f"Error during performance analysis for {self.config.strategy.symbol}"
                }

            performance_results['success'] = True
            logger.info(f"Backtest completed for {self.config.strategy.symbol} from {self.config.start_date} to {self.config.end_date}")
            return performance_results

        except Exception as e:
            logger.error(f"Error during backtesting: {e}", exc_info=True)
            raise BaseTradingException(f"Error during backtesting: {e}") from e
            return {
                'success': False,
                'message': f"Error during backtesting: {str(e)}"
            }

    @handle_error
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

        return self.visualization_service.plot_results(results, self.config.strategy.__class__.__name__, self.config.strategy.symbol, self.config.strategy.interval, filename)

    @handle_error
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

        return self.report_generator.generate_report(results, self.config.strategy.__class__.__name__, self.config.strategy.symbol, self.config.strategy.interval, filename)

    @staticmethod
    @handle_error
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
            raise BaseTradingException(f"Error generating comparison report: {e}") from e
            return f"Error generating comparison report: {e}"
