from database.database_connection import DatabaseConnection
from validation import validate_quantity, validate_risk


class PerformanceMetricsRepository:
    def __init__(self) -> None:
        self.db_connection = DatabaseConnection()

    async def __aenter__(self) -> "PerformanceMetricsRepository":
        await self.db_connection.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.db_connection.disconnect()

    async def insert_performance_metrics(
        self,
        trade_id: int,
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
    ) -> None:
        trade_id = int(validate_quantity(trade_id))
        initial_capital = validate_quantity(initial_capital)
        final_capital = validate_quantity(final_capital)
        total_return = validate_quantity(total_return)
        annual_return = validate_quantity(annual_return)
        max_drawdown = validate_quantity(max_drawdown)
        sharpe_ratio = validate_quantity(sharpe_ratio)
        win_rate = validate_risk(win_rate)
        avg_profit_pct = validate_quantity(avg_profit_pct)
        risk_reward_ratio = validate_quantity(risk_reward_ratio)
        profit_factor = validate_quantity(profit_factor)
        sql = """
            INSERT INTO performance_metrics (
                trade_id, initial_capital, final_capital, total_return,
                annual_return, max_drawdown, sharpe_ratio, win_rate,
                avg_profit_pct, risk_reward_ratio, profit_factor
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11
            )
        """
        values = (
            trade_id,
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
        )
        async with self.db_connection as conn:
            await conn.execute_query(sql, values)

    async def get_performance_metrics_by_trade_id(self, trade_id: int):
        trade_id = int(validate_quantity(trade_id))
        sql = """
            SELECT * FROM performance_metrics
            WHERE trade_id = $1
        """
        values = (trade_id,)
        async with self.db_connection as conn:
            return await conn.execute_query(sql, values)

