import logging
from typing import Dict, Any
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates
import pandas as pd

logger = logging.getLogger(__name__)

class VisualizationService:
    """
    Handles plotting functionality for backtest results.
    """

    def plot_results(self, results: Dict[str, Any], strategy_name: str, symbol: str, interval: str, filename: str = None) -> bool:
        """
        Plot the backtest results.

        Args:
            results (Dict[str, Any]): Backtest results from the PerformanceAnalyzer.
            strategy_name (str): Name of the strategy used in the backtest.
            symbol (str): Trading symbol used in the backtest.
            interval (str): Timeframe interval used in the backtest.
            filename (str, optional): Filename to save the plot. If None, the plot will be displayed.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not results:
                logger.error("Cannot plot results: No results provided.")
                return False

            data = results['data']
            metrics = results['metrics']
            trades = results['trades']

            # Create figure and grid
            fig = plt.figure(figsize=(12, 10))
            gs = GridSpec(3, 1, height_ratios=[2, 1, 1])

            # Price chart
            ax1 = plt.subplot(gs[0])
            ax1.set_title(f"{symbol} {interval} - {strategy_name} Backtest")
            ax1.plot(data.index, data['close'], linewidth=1, color='black', alpha=0.7, label='Price')

            # Add indicators based on strategy type
            if 'ema' in data.columns:
                ax1.plot(data.index, data['ema'], linewidth=1, color='blue', alpha=0.7, label='EMA')
            if 'vwap' in data.columns:
                ax1.plot(data.index, data['vwap'], linewidth=1, color='purple', alpha=0.7, label='VWAP')
            if 'fast_ema' in data.columns and 'slow_ema' in data.columns:
                ax1.plot(data.index, data['fast_ema'], linewidth=1, color='blue', alpha=0.7, label='Fast EMA')
                ax1.plot(data.index, data['slow_ema'], linewidth=1, color='orange', alpha=0.7, label='Slow EMA')

            # Add buy and sell markers
            buys = data[data['position_change'] > 0]
            sells = data[data['position_change'] < 0]

            ax1.scatter(buys.index, buys['close'], marker='^', color='green', s=50, label='Buy')
            ax1.scatter(sells.index, sells['close'], marker='v', color='red', s=50, label='Sell')

            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)

            # Set date format for x-axis
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax1.xaxis.set_major_locator(mdates.AutoDateLocator())

            # Portfolio value chart
            ax2 = plt.subplot(gs[1], sharex=ax1)
            ax2.set_title('Portfolio Value')
            ax2.plot(data.index, data['portfolio_value'], linewidth=1, color='green', label='Portfolio Value')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylabel('Value')

            # Add drawdown as a shaded area
            ax2.fill_between(data.index, data['portfolio_value'], data['peak'], color='red', alpha=0.3, label='Drawdown')

            # Strategy vs Buy and Hold
            ax3 = plt.subplot(gs[2], sharex=ax1)
            ax3.set_title('Strategy vs Buy and Hold')
            ax3.plot(data.index, data['strategy_cumulative_returns'], linewidth=1, color='blue', label='Strategy')
            ax3.plot(data.index, data['cumulative_returns'], linewidth=1, color='orange', label='Buy and Hold')
            ax3.grid(True, alpha=0.3)
            ax3.set_ylabel('Returns (%)')
            ax3.legend(loc='upper left')

            # Display key metrics as text
            textstr = '\n'.join((
                f"Total Return: {metrics['total_return_pct']:.2f}%",
                f"Annual Return: {metrics['annual_return_pct']:.2f}%",
                f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%",
                f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}",
                f"Trades: {metrics['total_trades']}",
                f"Win Rate: {metrics['win_rate_pct']:.2f}%",
                f"Profit Factor: {metrics['profit_factor']:.2f}"
            ))

            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=9,
                    verticalalignment='top', bbox=props)

            # Adjust layout
            plt.tight_layout()

            # Save or display the plot
            if filename:
                plt.savefig(filename)
                logger.info(f"Backtest plot saved as {filename}")
                plt.close(fig)
            else:
                plt.show()

            return True

        except Exception as e:
            logger.error(f"Error plotting results: {e}", exc_info=True)
            return False