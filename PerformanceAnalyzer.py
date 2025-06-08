import logging
import pandas as pd
from typing import Dict, Any
from utils import handle_error
from exceptions import BaseTradingException

logger = logging.getLogger(__name__)

class PerformanceAnalyzer:
    """
    Calculates performance metrics for backtesting results.
    """

    @handle_error
    def __init__(self, initial_capital: float = 10000.0, commission: float = 0.001):
        """
        Initializes the PerformanceAnalyzer with initial capital and commission rate.

        Args:
            initial_capital (float, optional): Initial capital for backtesting. Defaults to 10000.0.
            commission (float, optional): Commission rate per trade. Defaults to 0.001 (0.1%).
        """
        self.initial_capital = initial_capital
        self.commission = commission
        logger.info(f"Initialized PerformanceAnalyzer with initial capital: {initial_capital} and commission: {commission}")

    def calculate_performance(self, signals: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates performance metrics based on the backtesting results.

        Args:
            signals (pd.DataFrame): DataFrame with backtesting signals and close prices.

        Returns:
            Dict[str, Any]: Dictionary containing performance metrics.
        """
        try:
            # Prepare DataFrames for performance calculations
            backtest_results = signals.copy()

            # Calculate position changes (1 for buy, -1 for sell)
            backtest_results['position_change'] = backtest_results['signal'].diff().fillna(backtest_results['signal'])

            # Calculate returns
            backtest_results['returns'] = backtest_results['close'].pct_change().fillna(0)

            # Calculate strategy returns based on position
            backtest_results['strategy_returns'] = backtest_results['position'] * backtest_results['returns']

            # Apply commission costs to strategy returns when position changes
            backtest_results.loc[backtest_results['position_change'] != 0, 'strategy_returns'] -= self.commission

            # Calculate cumulative returns
            backtest_results['cumulative_returns'] = (1 + backtest_results['returns']).cumprod() - 1
            backtest_results['strategy_cumulative_returns'] = (1 + backtest_results['strategy_returns']).cumprod() - 1

            # Calculate portfolio value
            backtest_results['portfolio_value'] = self.initial_capital * (1 + backtest_results['strategy_cumulative_returns'])

            # Calculate drawdown
            backtest_results['peak'] = backtest_results['portfolio_value'].cummax()
            backtest_results['drawdown'] = (backtest_results['portfolio_value'] - backtest_results['peak']) / backtest_results['peak']

            # Calculate trade statistics
            trades = self._calculate_trades(backtest_results)

            # Calculate performance metrics
            metrics = self._calculate_metrics(backtest_results, trades, self.initial_capital)

            return {
                'data': backtest_results,
                'trades': trades,
                'metrics': metrics
            }
        except Exception as e:
            logger.error(f"Error during performance analysis: {e}", exc_info=True)
            raise BaseTradingException(f"Error during performance analysis: {e}") from e
            return {}

    @handle_error
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
                profit_pct -= self.commission  # Entry commission
                profit_pct -= self.commission  # Exit commission

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
            profit_pct -= self.commission

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

    @handle_error
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

        # Store performance metrics in the database
        from service.service_locator import ServiceLocator
        service_locator = ServiceLocator()
        performance_metrics_service = service_locator.get("PerformanceMetricsService")
        # Assuming you have trade_id available, replace 1 with the actual trade_id
        performance_metrics_service.insert_performance_metrics(
            trade_id=1,  # Replace with actual trade_id
            initial_capital=metrics['initial_capital'],
            final_capital=metrics['final_capital'],
            total_return=metrics['total_return'],
            annual_return=metrics['annual_return'],
            max_drawdown=metrics['max_drawdown'],
            sharpe_ratio=metrics['sharpe_ratio'],
            win_rate=metrics['win_rate'],
            avg_profit_pct=metrics['avg_profit_pct'],
            risk_reward_ratio=metrics['risk_reward_ratio'],
            profit_factor=metrics['profit_factor']
        )

        return metrics