from database.database_connection import DatabaseConnection

class RiskParametersRepository:
    def __init__(self):
        self.db_connection = DatabaseConnection()

    async def __aenter__(self):
        await self.db_connection.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.db_connection.disconnect()

    async def insert_risk_parameters(self, strategy_id, max_risk_per_trade, max_open_trades, stop_loss_percentage, take_profit_percentage, version):
        sql = """
            INSERT INTO risk_parameters (strategy_id, max_risk_per_trade, max_open_trades, stop_loss_percentage, take_profit_percentage, version)
            VALUES ($1, $2, $3, $4, $5, $6)
        """
        values = (strategy_id, max_risk_per_trade, max_open_trades, stop_loss_percentage, take_profit_percentage, version)
        await self.db_connection.execute_query(sql, values)

    async def get_risk_parameters_by_strategy_id(self, strategy_id):
        sql = """
            SELECT * FROM risk_parameters
            WHERE strategy_id = $1
        """
        values = (strategy_id,)
        return await self.db_connection.execute_query(sql, values)