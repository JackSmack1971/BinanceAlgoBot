from database.database_connection import DatabaseConnection
from validation import validate_risk, validate_quantity, sanitize_input


class RiskParametersRepository:
    def __init__(self) -> None:
        self.db_connection = DatabaseConnection()

    async def __aenter__(self) -> "RiskParametersRepository":
        await self.db_connection.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.db_connection.disconnect()

    async def insert_risk_parameters(
        self,
        strategy_id: int,
        max_risk_per_trade: float,
        max_open_trades: int,
        stop_loss_percentage: float,
        take_profit_percentage: float,
        version: str,
    ) -> None:
        strategy_id = int(validate_quantity(strategy_id))
        max_risk_per_trade = validate_risk(max_risk_per_trade)
        max_open_trades = int(validate_quantity(max_open_trades))
        stop_loss_percentage = validate_risk(stop_loss_percentage)
        take_profit_percentage = validate_risk(take_profit_percentage)
        version = sanitize_input(version)
        sql = """
            INSERT INTO risk_parameters (
                strategy_id, max_risk_per_trade, max_open_trades,
                stop_loss_percentage, take_profit_percentage, version
            ) VALUES ($1, $2, $3, $4, $5, $6)
        """
        values = (
            strategy_id,
            max_risk_per_trade,
            max_open_trades,
            stop_loss_percentage,
            take_profit_percentage,
            version,
        )
        async with self.db_connection as conn:
            await conn.execute_query(sql, values)

    async def get_risk_parameters_by_strategy_id(
        self, strategy_id: int, page_number: int = 1, page_size: int = 100
    ):
        strategy_id = int(validate_quantity(strategy_id))
        offset = (page_number - 1) * page_size
        sql = """
            SELECT * FROM risk_parameters
            WHERE strategy_id = $1
            LIMIT $2 OFFSET $3
        """
        values = (strategy_id, page_size, offset)
        async with self.db_connection as conn:
            return await conn.execute_query(sql, values)

