import sqlite3
import os

class SQLiteDatabase:
    def __init__(self):
        self.db_path = 'database/classifier.db'
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

    def create_table_with_composite_key(self, table_name, columns, primary_key_1, primary_key_2):
        """
        Create a new table in the database.
        
        Args:
        - table_name: Name of the table to be created.
        - columns: A dictionary where keys are column names and values are the data types.
                   Example: {'id': 'INTEGER PRIMARY KEY', 'name': 'TEXT', 'age': 'INTEGER'}
        """
        columns_str = ', '.join([f'{name} {data_type}' for name, data_type in columns.items()])
        create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_str}, PRIMARY KEY ({primary_key_1}, {primary_key_2}))"
        self.cursor.execute(create_table_query)
        self.conn.commit()

    def create_table_with_foreign_key(self, table_name, columns, foreign_key, foreign_table):
        """
        Create a new table in the database.

        Args:
        - table_name: Name of the table to be created.
        - columns: A dictionary where keys are column names and values are the data types.
                Example: {'id': 'INTEGER PRIMARY KEY', 'name': 'TEXT', 'age': 'INTEGER'}
        """
        columns_str = ', '.join([f'{name} {data_type}' for name, data_type in columns.items()])
        create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_str}, FOREIGN KEY ({foreign_key}) REFERENCES {foreign_table}({foreign_key}))"
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

    def view_table_structure(self):
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = self.cursor.fetchall()

        for table in tables:
            table_name = table[0]
            print(f"Table: {table_name}")
            print("----------")
            
            self.cursor.execute(f"PRAGMA table_info({table_name});")
            columns = self.cursor.fetchall()
            
            for column in columns:
                print(f"{column[1]} | {column[2]}")
            print()

    def get_class_info_table_data(self):
        """
        View data from the specified table.

        Args:
        - table_name: Name of the table.
        """
        select_query = f"SELECT * FROM class_info"
        self.cursor.execute(select_query)
        table_data = self.cursor.fetchall()

        data = []

        for row in table_data:
            data.append(row[0])

        return data

    def view_table_data(self, table_name):
        """
        View data from the specified table.

        Args:
        - table_name: Name of the table.
        """
        select_query = f"SELECT * FROM {table_name}"
        self.cursor.execute(select_query)
        table_data = self.cursor.fetchall()

        print(f"Data from table: {table_name}")
        print("----------")
        for row in table_data:
            print(row)

    def close_connection(self):
        """Close the database connection."""
        self.conn.close()

# # Example usage:
if __name__ == "__main__":
    # db_path = os.path.join(os.getcwd(), 'classifier.db')
    db = SQLiteDatabase()

#     db.create_table_with_foreign_key('model_info', {'datamodel_id': 'TEXT PRIMARY KEY', 'accuracy': 'REAL', 'f1_score': 'REAL', 'dataset_id' : 'TEXT'}, 'dataset_id', 'dataset_info')
#     db.create_table_with_composite_key('dataset_info', {'dataset_id': 'TEXT', 'data_id': 'TEXT', 'data': 'TEXT', 'label': 'TEXT'}, 'dataset_id', 'data_id')
#     db.view_table_structure()

#     # db.insert_record('models', {'datamodel_id': 'John Doe', 'accuracy': 30.253, 'f1_score': 45.225})
#     # db.insert_record('users', {'name': 'Jane Smith', 'age': 25})

#     # db.update_record('users', 1, {'name': 'John Updated', 'age': 35})

#     # db.delete_record('users', 2)

#     # result = db.execute_query("SELECT * FROM models")
#     # print(result)

    db.view_table_data('class_info')
    db.view_table_data('class_dataset_info')
    db.close_connection()