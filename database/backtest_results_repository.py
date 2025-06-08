from typing import Any

from database.database_connection import DatabaseConnection


class BacktestResultsRepository:
    def __init__(self) -> None:
        self.db_connection = DatabaseConnection()

    async def __aenter__(self) -> "BacktestResultsRepository":
        await self.db_connection.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.db_connection.disconnect()

    async def insert_backtest_results(
        self,
        strategy_id: int,
        start_date: Any,
        end_date: Any,
        initial_capital: float,
        final_capital: float,
        total_return: float,
        annual_return: float,
        max_drawdown: float,
        sharpe_ratio: float,
        win_rate: float,
        avg_profit_pct: float,
        risk_reward_ratio: float,
        profit_factor: float,
        total_trades: int,
        winning_trades: int,
        losing_trades: int,
    ) -> None:
        sql = """
            INSERT INTO backtest_results (
                strategy_id, start_date, end_date, initial_capital,
                final_capital, total_return, annual_return, max_drawdown,
                sharpe_ratio, win_rate, avg_profit_pct, risk_reward_ratio,
                profit_factor, total_trades, winning_trades, losing_trades
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8,
                $9, $10, $11, $12, $13, $14, $15, $16
            )
        """
        values = (
            strategy_id,
            start_date,
            end_date,
            initial_capital,
            final_capital,
            total_return,
            annual_return,
            max_drawdown,
            sharpe_ratio,
            win_rate,
            avg_profit_pct,
            risk_reward_ratio,
            profit_factor,
            total_trades,
            winning_trades,
            losing_trades,
        )
        async with self.db_connection as conn:
            await conn.execute_query(sql, values)

    async def get_backtest_results_by_strategy_id(self, strategy_id: int):
        sql = """
            SELECT * FROM backtest_results
            WHERE strategy_id = $1
        """
        values = (strategy_id,)
        async with self.db_connection as conn:
            return await conn.execute_query(sql, values)

