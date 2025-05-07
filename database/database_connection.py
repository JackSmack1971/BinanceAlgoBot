import psycopg2
from config import DATABASE_URL

class DatabaseConnection:
    def __init__(self):
        self.conn = None

    def connect(self):
        try:
            self.conn = psycopg2.connect(DATABASE_URL)
            print("Database connection established.")
        except psycopg2.Error as e:
            print(f"Error connecting to the database: {e}")

    def disconnect(self):
        if self.conn:
            self.conn.close()
            print("Database connection closed.")

    def get_connection(self):
        if not self.conn:
            self.connect()
        return self.conn