import logging
from typing import Any, Dict
from decimal import Decimal

from finance.financial_calculator import FinancialCalculator

import pandas as pd
from utils import handle_error
from exceptions import BaseTradingException

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """Calculate performance metrics for backtesting results."""

    @handle_error
    def __init__(self, initial_capital: float = 10000.0, commission: float = 0.001) -> None:
        self.calculator = FinancialCalculator()
        self.initial_capital = Decimal(str(initial_capital))
        self.commission = Decimal(str(commission))
        logger.info(
            "Initialized PerformanceAnalyzer with initial capital: %s and commission: %s",
            self.initial_capital,
            self.commission,
        )

    async def calculate_performance(self, signals: pd.DataFrame) -> Dict[str, Any]:
        try:
            results = signals.copy()
            results["position_change"] = results["signal"].diff().fillna(results["signal"])
            results["returns"] = results["close"].pct_change().fillna(0)
            results["strategy_returns"] = results["position"] * results["returns"]
            results.loc[results["position_change"] != 0, "strategy_returns"] -= self.commission
            results["cumulative_returns"] = (1 + results["returns"]).cumprod() - 1
            results["strategy_cumulative_returns"] = (1 + results["strategy_returns"]).cumprod() - 1
            results["portfolio_value"] = self.initial_capital * (1 + results["strategy_cumulative_returns"])
            results["peak"] = results["portfolio_value"].cummax()
            results["drawdown"] = (results["portfolio_value"] - results["peak"]) / results["peak"]

            trades = self._calculate_trades(results)
            metrics = await self._calculate_metrics(results, trades, self.initial_capital)
            return {"data": results, "trades": trades, "metrics": metrics}
        except Exception as exc:
            logger.error("Error during performance analysis: %s", exc, exc_info=True)
            raise BaseTradingException(f"Error during performance analysis: {exc}") from exc

    @handle_error
    def _calculate_trades(self, results: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame with individual trade statistics."""
        trades = []
        in_position = False
        entry_price = Decimal("0")
        entry_time = None
        position_type = None
        for idx, row in results.iterrows():
            if not in_position and row["position"] != 0:
                in_position = True
                entry_price = Decimal(str(row["close"]))
                entry_time = idx
                position_type = "long" if row["position"] > 0 else "short"
            elif in_position and (row["position"] == 0 or (row["position"] > 0 and position_type == "short") or (row["position"] < 0 and position_type == "long")):
                exit_price = Decimal(str(row["close"]))
                exit_time = idx
                pnl = self.calculator.calculate_profit_loss(entry_price, exit_price, Decimal("1"), "BUY" if position_type == "long" else "SELL")
                profit_pct = pnl["percentage_pnl"] - (self.commission * Decimal("200"))
                trades.append({
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "position_type": position_type,
                    "entry_price": float(entry_price),
                    "exit_price": float(exit_price),
                    "profit_pct": float(profit_pct),
                    "duration": (exit_time - entry_time).total_seconds() / 3600,
                })
                in_position = False
                if row["position"] != 0:
                    in_position = True
                    entry_price = Decimal(str(row["close"]))
                    entry_time = idx
                    position_type = "long" if row["position"] > 0 else "short"
        if in_position:
            exit_price = Decimal(str(results.iloc[-1]["close"]))
            exit_time = results.index[-1]
            pnl = self.calculator.calculate_profit_loss(entry_price, exit_price, Decimal("1"), "BUY" if position_type == "long" else "SELL")
            profit_pct = pnl["percentage_pnl"] - (self.commission * Decimal("100"))
            trades.append({
                "entry_time": entry_time,
                "exit_time": exit_time,
                "position_type": position_type,
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "profit_pct": float(profit_pct),
                "duration": (exit_time - entry_time).total_seconds() / 3600,
                "status": "open",
            })
        return pd.DataFrame(trades)

    @handle_error
    async def _calculate_metrics(self, results: pd.DataFrame, trades: pd.DataFrame, initial_capital: float) -> Dict[str, Any]:
        """Compute various performance metrics from trade results."""
        metrics: Dict[str, Any] = {}
        metrics["initial_capital"] = initial_capital
        metrics["final_capital"] = results["portfolio_value"].iloc[-1]
        metrics["total_return"] = metrics["final_capital"] / initial_capital - 1
        metrics["total_return_pct"] = metrics["total_return"] * 100
        days = (results.index[-1] - results.index[0]).days
        if days > 0:
            metrics["annual_return"] = (1 + metrics["total_return"]) ** (365 / days) - 1
            metrics["annual_return_pct"] = metrics["annual_return"] * 100
        else:
            metrics["annual_return"] = 0
            metrics["annual_return_pct"] = 0
        metrics["max_drawdown"] = results["drawdown"].min()
        metrics["max_drawdown_pct"] = metrics["max_drawdown"] * 100
        daily_returns = results["strategy_returns"].resample("D").sum()
        metrics["volatility"] = daily_returns.std() * (252 ** 0.5)
        metrics["sharpe_ratio"] = metrics["annual_return"] / metrics["volatility"] if metrics["volatility"] > 0 else 0
        if not trades.empty:
            metrics["total_trades"] = len(trades)
            metrics["winning_trades"] = (trades["profit_pct"] > 0).sum()
            metrics["losing_trades"] = (trades["profit_pct"] <= 0).sum()
            if metrics["total_trades"] > 0:
                metrics["win_rate"] = metrics["winning_trades"] / metrics["total_trades"]
                metrics["win_rate_pct"] = metrics["win_rate"] * 100
            else:
                metrics["win_rate"] = 0
                metrics["win_rate_pct"] = 0
            metrics["avg_profit_pct"] = trades["profit_pct"].mean() * 100
            metrics["avg_win_pct"] = trades.loc[trades["profit_pct"] > 0, "profit_pct"].mean() * 100 if metrics["winning_trades"] > 0 else 0
            metrics["avg_loss_pct"] = trades.loc[trades["profit_pct"] <= 0, "profit_pct"].mean() * 100 if metrics["losing_trades"] > 0 else 0
            metrics["risk_reward_ratio"] = abs(metrics["avg_win_pct"] / metrics["avg_loss_pct"]) if metrics["avg_loss_pct"] != 0 else 0
            metrics["avg_trade_duration_hours"] = trades["duration"].mean()
            total_gains = trades.loc[trades["profit_pct"] > 0, "profit_pct"].sum()
            total_losses = abs(trades.loc[trades["profit_pct"] <= 0, "profit_pct"].sum())
            metrics["profit_factor"] = total_gains / total_losses if total_losses > 0 else (float("inf") if total_gains > 0 else 0)
        else:
            metrics.update({
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "win_rate_pct": 0,
                "avg_profit_pct": 0,
                "avg_win_pct": 0,
                "avg_loss_pct": 0,
                "risk_reward_ratio": 0,
                "avg_trade_duration_hours": 0,
                "profit_factor": 0,
            })
        await self._store_metrics(metrics)
        return metrics

    async def _store_metrics(self, metrics: Dict[str, Any]) -> None:
        """Persist calculated metrics using the service layer."""
        from service.service_locator import ServiceLocator

        service_locator = ServiceLocator()
        performance_metrics_service = service_locator.get("PerformanceMetricsService")
        await performance_metrics_service.insert_performance_metrics(
            trade_id=1,
            initial_capital=metrics["initial_capital"],
            final_capital=metrics["final_capital"],
            total_return=metrics["total_return"],
            annual_return=metrics["annual_return"],
            max_drawdown=metrics["max_drawdown"],
            sharpe_ratio=metrics["sharpe_ratio"],
            win_rate=metrics.get("win_rate", 0),
            avg_profit_pct=metrics.get("avg_profit_pct", 0),
            risk_reward_ratio=metrics.get("risk_reward_ratio", 0),
            profit_factor=metrics.get("profit_factor", 0),
        )

