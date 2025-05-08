import logging
from typing import Dict, Any
import pandas as pd

from utils import handle_error
logger = logging.getLogger(__name__)

class ReportGenerator:
    """
    Generates a detailed report of the backtest results.
    """

    @handle_error
    def generate_report(self, results: Dict[str, Any], strategy_name: str, symbol: str, interval: str, filename: str = None) -> str:
        """
        Generates a detailed report of the backtest results.

        Args:
            results (Dict[str, Any]): Backtest results from the PerformanceAnalyzer.
            strategy_name (str): Name of the strategy used in the backtest.
            symbol (str): Trading symbol used in the backtest.
            interval (str): Timeframe interval used in the backtest.
            filename (str, optional): Filename to save the report. If None, the report will be returned as a string.

        Returns:
            str: Report as a string if filename is None, otherwise the path to the saved report
        """
        try:
            if not results:
                return "Cannot generate report: No results provided."

            metrics = results['metrics']
            trades = results['trades']
            data = results['data']

            # Generate report header
            report = f"# Backtest Report: {symbol} {interval} - {strategy_name}\n\n"

            # General information
            report += "## General Information\n\n"
            report += f"* Symbol: {symbol}\n"
            report += f"* Timeframe: {interval}\n"
            report += f"* Strategy: {strategy_name}\n"
            report += f"* Period: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}\n"
            report += f"* Duration: {(data.index[-1] - data.index[0]).days} days\n\n"

            # Performance metrics
            report += "## Performance Metrics\n\n"
            report += f"* Initial Capital: ${metrics['initial_capital']:.2f}\n"
            report += f"* Final Capital: ${metrics['final_capital']:.2f}\n"
            report += f"* Total Return: {metrics['total_return_pct']:.2f}%\n"
            report += f"* Annual Return: {metrics['annual_return_pct']:.2f}%\n"
            report += f"* Max Drawdown: {metrics['max_drawdown_pct']:.2f}%\n"
            report += f"* Volatility (annualized): {metrics['volatility'] * 100:.2f}%\n"
            report += f"* Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n\n"

            # Trade statistics
            report += "## Trade Statistics\n\n"
            report += f"* Total Trades: {metrics['total_trades']}\n"
            report += f"* Winning Trades: {metrics['winning_trades']} ({metrics['win_rate_pct']:.2f}%)\n"
            report += f"* Losing Trades: {metrics['losing_trades']} ({100 - metrics['win_rate_pct']:.2f}%)\n"
            report += f"* Average Profit/Loss: {metrics['avg_profit_pct']:.2f}%\n"
            report += f"* Average Win: {metrics['avg_win_pct']:.2f}%\n"
            report += f"* Average Loss: {metrics['avg_loss_pct']:.2f}%\n"
            report += f"* Risk-Reward Ratio: {metrics['risk_reward_ratio']:.2f}\n"
            report += f"* Profit Factor: {metrics['profit_factor']:.2f}\n"
            report += f"* Average Trade Duration: {metrics['avg_trade_duration_hours']:.2f} hours\n\n"

            # Strategy vs Buy and Hold
            buy_hold_return = data['cumulative_returns'].iloc[-1] * 100
            outperformance = metrics['total_return_pct'] - buy_hold_return

            report += "## Strategy vs Buy and Hold\n\n"
            report += f"* Strategy Return: {metrics['total_return_pct']:.2f}%\n"
            report += f"* Buy and Hold Return: {buy_hold_return:.2f}%\n"
            report += f"* Outperformance: {outperformance:.2f}%\n\n"

            # Monthly returns table
            monthly_returns = data['strategy_returns'].resample('M').apply(lambda x: (1 + x).prod() - 1)

            report += "## Monthly Returns\n\n"
            report += "| Month | Return (%) |\n"
            report += "| --- | --- |\n"

            for date, ret in monthly_returns.items():
                month_name = date.strftime('%b %Y')
                report += f"| {month_name} | {ret * 100:.2f}% |\n"

            report += "\n"

            # Save or return the report
            if filename:
                with open(filename, 'w') as f:
                    f.write(report)
                logger.info(f"Backtest report saved as {filename}")
                return filename
            else:
                return report

        except Exception as e:
            logger.error(f"Error generating report: {e}", exc_info=True)
            raise BaseTradingException(f"Error generating report: {e}") from e
            return f"Error generating report: {e}"