from database.database_connection import DatabaseConnection

class ConfigurationRepository:
    def __init__(self):
        self.db_connection = DatabaseConnection()

    def insert_configuration(self, config_name, config_value, version):
        conn = self.db_connection.get_connection()
        cursor = conn.cursor()
        sql = """
            INSERT INTO configurations (config_name, config_value, version)
            VALUES (%s, %s, %s)
        """
        values = (config_name, config_value, version)
        cursor.execute(sql, values)
        conn.commit()
        cursor.close()

    def get_configuration_by_name(self, config_name):
        conn = self.db_connection.get_connection()
        cursor = conn.cursor()
        sql = """
            SELECT * FROM configurations
            WHERE config_name = %s
        """
        values = (config_name,)
        cursor.execute(sql, values)
        results = cursor.fetchall()
        cursor.close()
        return results