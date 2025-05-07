from database.database_connection import DatabaseConnection

class RiskParametersRepository:
    def __init__(self):
        self.db_connection = DatabaseConnection()

    def insert_risk_parameters(self, strategy_id, max_risk_per_trade, max_open_trades, stop_loss_percentage, take_profit_percentage, version):
        conn = self.db_connection.get_connection()
        cursor = conn.cursor()
        sql = """
            INSERT INTO risk_parameters (strategy_id, max_risk_per_trade, max_open_trades, stop_loss_percentage, take_profit_percentage, version)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        values = (strategy_id, max_risk_per_trade, max_open_trades, stop_loss_percentage, take_profit_percentage, version)
        cursor.execute(sql, values)
        conn.commit()
        cursor.close()

    def get_risk_parameters_by_strategy_id(self, strategy_id):
        conn = self.db_connection.get_connection()
        cursor = conn.cursor()
        sql = """
            SELECT * FROM risk_parameters
            WHERE strategy_id = %s
        """
        values = (strategy_id,)
        cursor.execute(sql, values)
        results = cursor.fetchall()
        cursor.close()
        return results