import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Type, Tuple
from datetime import datetime
from binance.client import Client

from strategy_factory import Strategy
from config import TRADING_CONFIG, RISK_CONFIG

logger = logging.getLogger(__name__)

class Backtester:
    """
    Class for backtesting trading strategies using historical data.
    """
    
    def __init__(self, client: Client, strategy: Strategy):
        """
        Initialize the backtester.
        
        Args:
            client (Client): Binance API client
            strategy (Strategy): Trading strategy to backtest
        """
        self.client = client
        self.strategy = strategy
        
        logger.info(f"Initialized backtester for {strategy.__class__.__name__} on {strategy.symbol}/{strategy.interval}")
    
    def run(self, start_date: str, end_date: str, initial_capital: float = 10000.0, 
            commission: float = 0.001) -> Dict[str, Any]:
        """
        Run the backtest for the specified period.
        
        Args:
            start_date (str): Start date for backtesting (format: 'YYYY-MM-DD')
            end_date (str): End date for backtesting (format: 'YYYY-MM-DD')
            initial_capital (float, optional): Initial capital for backtesting. Defaults to 10000.0.
            commission (float, optional): Commission rate per trade. Defaults to 0.001 (0.1%).
            
        Returns:
            Dict[str, Any]: Backtesting results
        """
        try:
            logger.info(f"Running backtest from {start_date} to {end_date}")
            
            # Convert dates to milliseconds timestamp for Binance API
            start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
            end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
            
            # Get historical data from Binance
            klines = self.client.get_historical_klines(
                symbol=self.strategy.symbol,
                interval=self.strategy.interval,
                start_str=start_ts,
                end_str=end_ts
            )
            
            if not klines:
                logger.error(f"No historical data available for {self.strategy.symbol} from {start_date} to {end_date}")
                return {
                    'success': False,
                    'message': f"No historical data available for {self.strategy.symbol} from {start_date} to {end_date}"
                }
            
            # Convert to DataFrame
            data = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'trades',
                'taker_buy_base', 'taker_buy_quote', 'ignored'
            ])
            data = data.astype(float)
            
            # Convert timestamp to datetime
            data['datetime'] = pd.to_datetime(data['timestamp'], unit='ms')
            data.set_index('datetime', inplace=True)
            
            # Apply the strategy to generate signals
            # We'll override the strategy's data_feed to use our historical data
            self.strategy.data_feed.get_data = lambda: data
            
            # Calculate indicators and generate signals
            signals = self.strategy.run()
            
            if signals is None or signals.empty:
                logger.error(f"No signals generated for {self.strategy.symbol}")
                return {
                    'success': False,
                    'message': f"No signals generated for {self.strategy.symbol}"
                }
            
            # Prepare DataFrames for performance calculations
            # Make a copy to avoid modifying the original signals
            backtest_results = signals.copy()
            
            # Calculate position changes (1 for buy, -1 for sell)
            backtest_results['position_change'] = backtest_results['signal'].diff().fillna(backtest_results['signal'])
            
            # Calculate returns
            backtest_results['returns'] = backtest_results['close'].pct_change().fillna(0)
            
            # Calculate strategy returns based on position
            backtest_results['strategy_returns'] = backtest_results['position'] * backtest_results['returns']
            
            # Apply commission costs to strategy returns when position changes
            backtest_results.loc[backtest_results['position_change'] != 0, 'strategy_returns'] -= commission
            
            # Calculate cumulative returns
            backtest_results['cumulative_returns'] = (1 + backtest_results['returns']).cumprod() - 1
            backtest_results['strategy_cumulative_returns'] = (1 + backtest_results['strategy_returns']).cumprod() - 1
            
            # Calculate portfolio value
            backtest_results['portfolio_value'] = initial_capital * (1 + backtest_results['strategy_cumulative_returns'])
            
            # Calculate drawdown
            backtest_results['peak'] = backtest_results['portfolio_value'].cummax()
            backtest_results['drawdown'] = (backtest_results['portfolio_value'] - backtest_results['peak']) / backtest_results['peak']
            
            # Calculate trade statistics
            trades = self._calculate_trades(backtest_results)
            
            # Calculate performance metrics
            metrics = self._calculate_metrics(backtest_results, trades, initial_capital)
            
            logger.info(f"Backtest completed for {self.strategy.symbol} from {start_date} to {end_date}")
            
            return {
                'success': True,
                'data': backtest_results,
                'trades': trades,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Error during backtesting: {e}", exc_info=True)
            return {
                'success': False,
                'message': f"Error during backtesting: {str(e)}"
            }
    
    def _calculate_trades(self, results: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate individual trades and their statistics.
        
        Args:
            results (pd.DataFrame): DataFrame with backtest results
            
        Returns:
            pd.DataFrame: DataFrame with trade statistics
        """
        trades = []
        in_position = False
        entry_price = 0
        entry_time = None
        position_type = None
        
        for idx, row in results.iterrows():
            # Entry signal
            if not in_position and row['position'] != 0:
                in_position = True
                entry_price = row['close']
                entry_time = idx
                position_type = 'long' if row['position'] > 0 else 'short'
            
            # Exit signal or position change
            elif in_position and (row['position'] == 0 or (row['position'] > 0 and position_type == 'short') or (row['position'] < 0 and position_type == 'long')):
                exit_price = row['close']
                exit_time = idx
                
                # Calculate profit/loss
                if position_type == 'long':
                    profit_pct = (exit_price / entry_price) - 1
                else:  # short
                    profit_pct = 1 - (exit_price / entry_price)
                
                # Apply commission
                profit_pct -= 0.001  # Entry commission
                profit_pct -= 0.001  # Exit commission
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'position_type': position_type,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'profit_pct': profit_pct,
                    'duration': (exit_time - entry_time).total_seconds() / 3600  # Duration in hours
                })
                
                # Reset flags
                in_position = False
                
                # Check if we're entering a new position immediately
                if row['position'] != 0:
                    in_position = True
                    entry_price = row['close']
                    entry_time = idx
                    position_type = 'long' if row['position'] > 0 else 'short'
        
        # Handle open position at the end of the backtest period
        if in_position:
            exit_price = results.iloc[-1]['close']
            exit_time = results.index[-1]
            
            # Calculate profit/loss
            if position_type == 'long':
                profit_pct = (exit_price / entry_price) - 1
            else:  # short
                profit_pct = 1 - (exit_price / entry_price)
            
            # Apply commission (only entry, since we're simulating the exit)
            profit_pct -= 0.001
            
            trades.append({
                'entry_time': entry_time,
                'exit_time': exit_time,
                'position_type': position_type,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'profit_pct': profit_pct,
                'duration': (exit_time - entry_time).total_seconds() / 3600,
                'status': 'open'
            })
        
        return pd.DataFrame(trades)
    
    def _calculate_metrics(self, results: pd.DataFrame, trades: pd.DataFrame, initial_capital: float) -> Dict[str, Any]:
        """
        Calculate performance metrics for the backtest.
        
        Args:
            results (pd.DataFrame): DataFrame with backtest results
            trades (pd.DataFrame): DataFrame with trade statistics
            initial_capital (float): Initial capital for the backtest
            
        Returns:
            Dict[str, Any]: Dictionary with performance metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['initial_capital'] = initial_capital
        metrics['final_capital'] = results['portfolio_value'].iloc[-1]
        metrics['total_return'] = metrics['final_capital'] / initial_capital - 1
        metrics['total_return_pct'] = metrics['total_return'] * 100
        
        # Annualized return
        days = (results.index[-1] - results.index[0]).days
        if days > 0:
            metrics['annual_return'] = (1 + metrics['total_return']) ** (365 / days) - 1
            metrics['annual_return_pct'] = metrics['annual_return'] * 100
        else:
            metrics['annual_return'] = 0
            metrics['annual_return_pct'] = 0
        
        # Risk metrics
        metrics['max_drawdown'] = results['drawdown'].min()
        metrics['max_drawdown_pct'] = metrics['max_drawdown'] * 100
        
        # Volatility and risk-adjusted returns
        daily_returns = results['strategy_returns'].resample('D').sum()
        metrics['volatility'] = daily_returns.std() * (252 ** 0.5)  # Annualized
        
        if metrics['volatility'] > 0:
            metrics['sharpe_ratio'] = metrics['annual_return'] / metrics['volatility']
        else:
            metrics['sharpe_ratio'] = 0
        
        # Trade metrics
        if not trades.empty:
            metrics['total_trades'] = len(trades)
            metrics['winning_trades'] = (trades['profit_pct'] > 0).sum()
            metrics['losing_trades'] = (trades['profit_pct'] <= 0).sum()
            
            if metrics['total_trades'] > 0:
                metrics['win_rate'] = metrics['winning_trades'] / metrics['total_trades']
                metrics['win_rate_pct'] = metrics['win_rate'] * 100
            else:
                metrics['win_rate'] = 0
                metrics['win_rate_pct'] = 0
            
            # Average metrics
            metrics['avg_profit_pct'] = trades['profit_pct'].mean() * 100
            metrics['avg_win_pct'] = trades.loc[trades['profit_pct'] > 0, 'profit_pct'].mean() * 100 if metrics['winning_trades'] > 0 else 0
            metrics['avg_loss_pct'] = trades.loc[trades['profit_pct'] <= 0, 'profit_pct'].mean() * 100 if metrics['losing_trades'] > 0 else 0
            
            # Risk-reward
            if metrics['avg_loss_pct'] != 0:
                metrics['risk_reward_ratio'] = abs(metrics['avg_win_pct'] / metrics['avg_loss_pct'])
            else:
                metrics['risk_reward_ratio'] = 0
            
            # Average trade duration
            metrics['avg_trade_duration_hours'] = trades['duration'].mean()
            
            # Profit factor
            total_gains = trades.loc[trades['profit_pct'] > 0, 'profit_pct'].sum()
            total_losses = abs(trades.loc[trades['profit_pct'] <= 0, 'profit_pct'].sum())
            
            if total_losses > 0:
                metrics['profit_factor'] = total_gains / total_losses
            else:
                metrics['profit_factor'] = float('inf') if total_gains > 0 else 0
        else:
            metrics['total_trades'] = 0
            metrics['winning_trades'] = 0
            metrics['losing_trades'] = 0
            metrics['win_rate'] = 0
            metrics['win_rate_pct'] = 0
            metrics['avg_profit_pct'] = 0
            metrics['avg_win_pct'] = 0
            metrics['avg_loss_pct'] = 0
            metrics['risk_reward_ratio'] = 0
            metrics['avg_trade_duration_hours'] = 0
            metrics['profit_factor'] = 0
        
        return metrics
    
    def plot_results(self, results: Dict[str, Any], filename: str = None) -> bool:
        """
        Plot the backtest results.
        
        Args:
            results (Dict[str, Any]): Backtest results from the run method
            filename (str, optional): Filename to save the plot. If None, the plot will be displayed.
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
            import matplotlib.dates as mdates
            
            if not results['success']:
                logger.error(f"Cannot plot results: {results['message']}")
                return False
            
            data = results['data']
            metrics = results['metrics']
            trades = results['trades']
            
            # Create figure and grid
            fig = plt.figure(figsize=(12, 10))
            gs = GridSpec(3, 1, height_ratios=[2, 1, 1])
            
            # Price chart
            ax1 = plt.subplot(gs[0])
            ax1.set_title(f"{self.strategy.symbol} {self.strategy.interval} - {self.strategy.__class__.__name__} Backtest")
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
    
    def generate_report(self, results: Dict[str, Any], filename: str = None) -> str:
        """
        Generate a detailed report of the backtest results.
        
        Args:
            results (Dict[str, Any]): Backtest results from the run method
            filename (str, optional): Filename to save the report. If None, the report will be returned as a string.
            
        Returns:
            str: Report as a string if filename is None, otherwise the path to the saved report
        """
        try:
            if not results['success']:
                return f"Cannot generate report: {results['message']}"
            
            metrics = results['metrics']
            trades = results['trades']
            
            # Generate report header
            report = f"# Backtest Report: {self.strategy.symbol} {self.strategy.interval} - {self.strategy.__class__.__name__}\n\n"
            
            # General information
            report += "## General Information\n\n"
            report += f"* Symbol: {self.strategy.symbol}\n"
            report += f"* Timeframe: {self.strategy.interval}\n"
            report += f"* Strategy: {self.strategy.__class__.__name__}\n"
            report += f"* Period: {results['data'].index[0].strftime('%Y-%m-%d')} to {results['data'].index[-1].strftime('%Y-%m-%d')}\n"
            report += f"* Duration: {(results['data'].index[-1] - results['data'].index[0]).days} days\n\n"
            
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
            buy_hold_return = results['data']['cumulative_returns'].iloc[-1] * 100
            outperformance = metrics['total_return_pct'] - buy_hold_return
            
            report += "## Strategy vs Buy and Hold\n\n"
            report += f"* Strategy Return: {metrics['total_return_pct']:.2f}%\n"
            report += f"* Buy and Hold Return: {buy_hold_return:.2f}%\n"
            report += f"* Outperformance: {outperformance:.2f}%\n\n"
            
            # Monthly returns table
            monthly_returns = results['data']['strategy_returns'].resample('M').apply(lambda x: (1 + x).prod() - 1)
            
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
            return f"Error generating report: {str(e)}"
    
    @staticmethod
    def compare_strategies(backtests: List[Dict[str, Any]], filename: str = None) -> Optional[str]:
        """
        Compare multiple backtest results.
        
        Args:
            backtests (List[Dict[str, Any]]): List of backtest results from the run method
            filename (str, optional): Filename to save the comparison report. If None, the report will be returned as a string.
            
        Returns:
            Optional[str]: Comparison report as a string if filename is None, otherwise the path to the saved report
        """
        try:
            if not backtests:
                return "No backtest results to compare"
            
            # Generate report header
            report = "# Strategy Comparison Report\n\n"
            
            # Performance comparison table
            report += "## Performance Comparison\n\n"
            report += "| Strategy | Symbol | Timeframe | Total Return (%) | Annual Return (%) | Max Drawdown (%) | Sharpe Ratio | Trades | Win Rate (%) |\n"
            report += "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
            
            for backtest in backtests:
                if not backtest['success']:
                    continue
                
                metrics = backtest['metrics']
                strategy_name = backtest.get('strategy_name', 'Unknown')
                symbol = backtest.get('symbol', 'Unknown')
                interval = backtest.get('interval', 'Unknown')
                
                report += f"| {strategy_name} | {symbol} | {interval} | "
                report += f"{metrics['total_return_pct']:.2f} | "
                report += f"{metrics['annual_return_pct']:.2f} | "
                report += f"{metrics['max_drawdown_pct']:.2f} | "
                report += f"{metrics['sharpe_ratio']:.2f} | "
                report += f"{metrics['total_trades']} | "
                report += f"{metrics['win_rate_pct']:.2f} |\n"
            
            report += "\n"
            
            # Risk-reward comparison
            report += "## Risk-Reward Comparison\n\n"
            report += "| Strategy | Profit Factor | Risk-Reward Ratio | Avg Profit (%) | Avg Win (%) | Avg Loss (%) |\n"
            report += "| --- | --- | --- | --- | --- | --- |\n"
            
            for backtest in backtests:
                if not backtest['success']:
                    continue
                
                metrics = backtest['metrics']
                strategy_name = backtest.get('strategy_name', 'Unknown')
                
                report += f"| {strategy_name} | "
                report += f"{metrics['profit_factor']:.2f} | "
                report += f"{metrics['risk_reward_ratio']:.2f} | "
                report += f"{metrics['avg_profit_pct']:.2f} | "
                report += f"{metrics['avg_win_pct']:.2f} | "
                report += f"{metrics['avg_loss_pct']:.2f} |\n"
            
            report += "\n"
            
            # Consistency comparison
            report += "## Trading Consistency\n\n"
            report += "| Strategy | Total Trades | Avg Duration (h) | Trades per Month |\n"
            report += "| --- | --- | --- | --- |\n"
            
            for backtest in backtests:
                if not backtest['success']:
                    continue
                
                metrics = backtest['metrics']
                data = backtest['data']
                strategy_name = backtest.get('strategy_name', 'Unknown')
                
                # Calculate trades per month
                days = (data.index[-1] - data.index[0]).days
                if days > 0:
                    trades_per_month = metrics['total_trades'] / (days / 30)
                else:
                    trades_per_month = 0
                
                report += f"| {strategy_name} | "
                report += f"{metrics['total_trades']} | "
                report += f"{metrics['avg_trade_duration_hours']:.2f} | "
                report += f"{trades_per_month:.2f} |\n"
            
            report += "\n"
            
            # Save or return the report
            if filename:
                with open(filename, 'w') as f:
                    f.write(report)
                logger.info(f"Strategy comparison report saved as {filename}")
                return filename
            else:
                return report
            
        except Exception as e:
            logger.error(f"Error comparing strategies: {e}", exc_info=True)
            return f"Error comparing strategies: {str(e)}"
