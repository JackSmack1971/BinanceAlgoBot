from database.database_connection import DatabaseConnection

class ConfigurationRepository:
    def __init__(self):
        self.db_connection = DatabaseConnection()

    async def __aenter__(self):
        await self.db_connection.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.db_connection.disconnect()

    async def insert_configuration(self, config_name, config_value, version):
        sql = """
            INSERT INTO configurations (config_name, config_value, version)
            VALUES ($1, $2, $3)
        """
        values = (config_name, config_value, version)
        await self.db_connection.execute_query(sql, values)

    async def get_configuration_by_name(self, config_name):
        sql = """
            SELECT * FROM configurations
            WHERE config_name = $1
        """
        values = (config_name,)
        return await self.db_connection.execute_query(sql, values)