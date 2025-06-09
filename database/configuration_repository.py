from database.database_connection import DatabaseConnection
from validation import sanitize_input


class ConfigurationRepository:
    def __init__(self) -> None:
        self.db_connection = DatabaseConnection()

    async def __aenter__(self) -> "ConfigurationRepository":
        await self.db_connection.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.db_connection.disconnect()

    async def insert_configuration(self, config_name: str, config_value: str, version: str) -> None:
        config_name = sanitize_input(config_name)
        config_value = sanitize_input(config_value)
        version = sanitize_input(version)
        sql = """
            INSERT INTO configurations (config_name, config_value, version)
            VALUES ($1, $2, $3)
        """
        values = (config_name, config_value, version)
        async with self.db_connection as conn:
            await conn.execute_query(sql, values)

    async def get_configuration_by_name(self, config_name: str):
        config_name = sanitize_input(config_name)
        sql = """
            SELECT * FROM configurations
            WHERE config_name = $1
        """
        values = (config_name,)
        async with self.db_connection as conn:
            return await conn.execute_query(sql, values)

