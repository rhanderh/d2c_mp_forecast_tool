import psycopg2
import psycopg2.extras as extras
import pandas as pd
from psycopg2 import pool
import streamlit as st
from typing import List, Dict, Union, Tuple


class PostgreSQLManager:
    def __init__(self):
        """
        Initialize the PostgreSQLManager by loading the database configuration
        and creating a connection pool.
        """
        self.database_config = self.load_database_config_from_secrets()
        self.database_pool = self.create_connection_pool()

    def load_database_config_from_secrets(self) -> Dict[str, Union[str, int]]:
        """
        Load the database configuration from the secrets.toml file.

        Returns:
            Dict[str, Union[str, int]]: A dictionary containing the database configuration.
        """
        try:
            return {
                "user": st.secrets["postgresql"]["user"],
                "password": st.secrets["postgresql"]["password"],
                "host": st.secrets["postgresql"]["host"],
                "port": st.secrets["postgresql"]["port"],
                "database": st.secrets["postgresql"]["database"],
            }
        except KeyError as e:
            st.error(
                f"Error loading database configuration from secrets.toml: {e}")
            return {}

    def create_connection_pool(self) -> psycopg2.pool.SimpleConnectionPool:
        """
        Create a connection pool for the PostgreSQL database.

        Returns:
            psycopg2.pool.SimpleConnectionPool: A psycopg2 SimpleConnectionPool instance, or None if an error occurs.
        """
        try:
            database_pool = psycopg2.pool.SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                user=self.database_config["user"],
                password=self.database_config["password"],
                host=self.database_config["host"],
                port=self.database_config["port"],
                database=self.database_config["database"],
            )
            return database_pool
        except Exception as e:
            st.error(f"Error connecting to the database: {e}")
            return None

    def execute_query(self, query: str, params: Union[List[Union[str, int, float]], None] = None) -> List[Dict]:
        """
        Execute the given query with optional parameters, and return the results.

        Args:
            query (str): The SQL query to execute.
            params (Union[List[Union[str, int, float]], None], optional): A list of query parameters. Defaults to None.

        Returns:
            List[Dict]: A list of dictionaries containing the results of the query.
        """
        connection = self.database_pool.getconn()
        try:
            with connection.cursor() as cursor:
                cursor.execute(query, params)
                connection.commit()
                column_names = [desc[0] for desc in cursor.description]
                result = [dict(zip(column_names, row))
                          for row in cursor.fetchall()]
                return result
        except Exception as e:
            st.error(f"Error executing query: {e}")
            return []
        finally:
            self.database_pool.putconn(connection)

    def insert_data(self, query: str, params: Union[Tuple[Union[str, int, float]], None] = None) -> bool:
        """
        Insert data into the database using the given query and parameters.

        Args:
            query (str): The SQL query for inserting data.
            params (Union[Tuple[Union[str, int, float]], None], optional): A tuple of query parameters. Defaults to None.

        Returns:
            bool: True if the insertion is successful, False otherwise.
        """
        connection = self.database_pool.getconn()
        try:
            with connection.cursor() as cursor:
                cursor.execute(query, params)
                connection.commit()
                return True
        except Exception as e:
            st.error(f"Error inserting data: {e}")
            return False
        finally:
            self.database_pool.putconn(connection)

    def insert_dataframe(self, dataframe: pd.DataFrame, table_name: str) -> bool:
        """
        Insert data from a DataFrame into the specified table in the database.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing data to be inserted.
            table_name (str): The name of the table in the database.

        Returns:
            bool: True if the insertion is successful, False otherwise.
        """
        connection = self.database_pool.getconn()
        try:
            with connection.cursor() as cursor:
                columns = ", ".join([f'"{col}"' for col in dataframe.columns])
                values_template = ", ".join(
                    [f"%({col})s" for col in dataframe.columns])
                insert_query = f'INSERT INTO {table_name} ({columns}) VALUES ({values_template})'
                data_dicts = dataframe.to_dict('records')
                extras.execute_batch(cursor, insert_query, data_dicts)
                connection.commit()
                return True
        except Exception as e:
            st.error(f"Error inserting DataFrame data: {e}")
            return False
        finally:
            self.database_pool.putconn(connection)

    def close_pool(self):
        self.database_pool.closeall()

# Example usage:
# if __name__ == "__main__":
    # Initialize the PostgreSQLManager
    # pg_manager = PostgreSQLManager()

    # Sample query
    # result = pg_manager.execute_query("SELECT * FROM your_table_name;")
    # print(result)

    # Close the connection pool
    # pg_manager.close_pool()
