import sqlite3
import os

class SQLiteDatabase:
    def __init__(self):
        self.db_path = 'classifier.db'
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

    def create_table(self, table_name, columns):
        """
        Create a new table in the database.
        
        Args:
        - table_name: Name of the table to be created.
        - columns: A dictionary where keys are column names and values are the data types.
                   Example: {'id': 'INTEGER PRIMARY KEY', 'name': 'TEXT', 'age': 'INTEGER'}
        """
        columns_str = ', '.join([f'{name} {data_type}' for name, data_type in columns.items()])
        create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_str})"
        self.cursor.execute(create_table_query)
        self.conn.commit()

    def insert_record(self, table_name, record):
        """
        Insert a new record into the specified table.
        
        Args:
        - table_name: Name of the table.
        - record: A dictionary representing the record to be inserted.
                  Example: {'name': 'John Doe', 'age': 30, 'email': 'john@example.com'}
        """
        columns = ', '.join(record.keys())
        placeholders = ', '.join(['?' for _ in record.keys()])
        insert_query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        self.cursor.execute(insert_query, list(record.values()))
        self.conn.commit()

    def update_record(self, table_name, record_id, new_values):
        """
        Update an existing record in the specified table.
        
        Args:
        - table_name: Name of the table.
        - record_id: Identifier of the record to be updated.
        - new_values: A dictionary representing the new values for the record.
        """
        set_clause = ', '.join([f'{key} = ?' for key in new_values.keys()])
        update_query = f"UPDATE {table_name} SET {set_clause} WHERE id = ?"
        self.cursor.execute(update_query, list(new_values.values()) + [record_id])
        self.conn.commit()

    def delete_record(self, table_name, record_id):
        """
        Delete an existing record from the specified table.
        
        Args:
        - table_name: Name of the table.
        - record_id: Identifier of the record to be deleted.
        """
        delete_query = f"DELETE FROM {table_name} WHERE id = ?"
        self.cursor.execute(delete_query, (record_id,))
        self.conn.commit()

    def delete_all_records(self, table_name):
        """
        Delete all records from the specified table.
        
        Args:
        - table_name: Name of the table.
        """
        delete_query = f"DELETE FROM {table_name}"
        self.cursor.execute(delete_query)
        self.conn.commit()

    def execute_query(self, query):
        """
        Execute a custom SQL query.
        
        Args:
        - query: The SQL query to be executed.
        
        Returns:
        - result: The result of the query execution.
        """
        self.cursor.execute(query)
        result = self.cursor.fetchall()
        self.conn.commit()
        return result

    def close_connection(self):
        """Close the database connection."""
        self.conn.close()

# Example usage:
if __name__ == "__main__":
    # db_path = os.path.join(os.getcwd(), 'classifier.db')
    db = SQLiteDatabase()

    db.create_table('models', {'model_id': 'TEXT PRIMARY KEY', 'accuracy': 'REAL', 'f1_score': 'REAL'})

    # db.insert_record('models', {'model_id': 'John Doe', 'accuracy': 30.253, 'f1_score': 45.225})
    # db.insert_record('users', {'name': 'Jane Smith', 'age': 25})

    # db.update_record('users', 1, {'name': 'John Updated', 'age': 35})

    # db.delete_record('users', 2)

    result = db.execute_query("SELECT * FROM models")
    print(result)

    db.close_connection()
