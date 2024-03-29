import csv
import json
import yaml
from time import time
from database.database_connection import SQLiteDatabase

class DataImporter:
    def __init__(self):
        pass

    def import_data_from_file(self, dataset_name, file_location, upload_class_name = False):
        try:
            data, file_type = self._load_data(file_location)

            if data:
                dataset_id = int(time())
                if (file_type == "json" or "csv") and type(data)==dict:
                    data_point_count = 0
                    for class_name, values in data.items():
                        for string_value in values:
                            record = {
                                'dataset_id': str(dataset_id),
                                'data_id': str(data_point_count),
                                'data': str(string_value),
                                'class_name': str(class_name).upper()
                            }
                            data_point_count += 1
                            SQLiteDatabase().insert_record('data_info', record)
                            query = f"""INSERT OR IGNORE INTO class_info (class_name) VALUES ('{str(class_name).upper()}')"""
                            SQLiteDatabase().execute_query(query)

                            query = f"""INSERT OR IGNORE INTO class_dataset_info (class_name, dataset_id) VALUES ('{str(class_name).upper()}','{str(dataset_id)}')"""
                            SQLiteDatabase().execute_query(query)
                    SQLiteDatabase().insert_record('dataset_info', {'dataset_id': str(dataset_id),'dataset_name':str(dataset_name)})
                else:
                    if upload_class_name:
                        for idx, datum in enumerate(data):
                            record = {
                                'dataset_id': str(dataset_id),
                                'data_id': str(idx),
                                'data': str(datum),
                                'class_name': str(upload_class_name).upper()
                            }
                            SQLiteDatabase().insert_record('data_info', record)
                            query = f"""INSERT OR IGNORE INTO class_info (class_name) VALUES ('{str(upload_class_name).upper()}')"""
                            SQLiteDatabase().execute_query(query)
                            query = f"""INSERT OR IGNORE INTO class_dataset_info (class_name, dataset_id) VALUES ('{str(upload_class_name).upper()}','{str(dataset_id)}')"""
                            SQLiteDatabase().execute_query(query)
                    else:
                        for idx, datum in enumerate(data):
                            record = {
                                'dataset_id': str(dataset_id),
                                'data_id': str(idx),
                                'data': str(datum),
                                'class_name': ''
                            }
                            SQLiteDatabase().insert_record('data_info', record)
                    SQLiteDatabase().insert_record('dataset_info', {'dataset_id': str(dataset_id),'dataset_name':str(dataset_name)})
            return True
        except Exception as e:
            self.handle_error('_load_data', e)

    def _load_data(self, file_location):
        try:
            if file_location.endswith('.json'):
                return (self._load_json(file_location), "json")
            elif file_location.endswith('.csv'):
                return (self._load_csv(file_location), "csv")
            elif file_location.endswith('.yaml'):
                return self._load_yaml((file_location), "yaml")
            else:
                print("Unsupported file format")
                return False
        except Exception as e:
            self.handle_error('_load_data', e)

    def _load_json(self, file_location):
        try:
            with open(file_location, 'r') as file:
                data = json.load(file)
            return data
        except Exception as e:
            self.handle_error('_load_json', e)

    def _load_csv(self, file_location):
        try:
            data = {}
            data_list = []
            state  = False
            with open(file_location, 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    if len(row) == 1:  # If only one column
                        data_list.append(row[0])
                        state = True
                    elif len(row) == 2:  # If two columns
                        key = row[1]
                        value = row[0]
                        if key not in data:
                            data[key] = []
                        data[key] = value
            if state:
                return data_list
            else:
                return data
        except Exception as e:
            self.handle_error('_load_csv', e)

    def _load_yaml(self, file_location):
        try:
            with open(file_location, 'r') as file:
                data = yaml.safe_load(file)
            return data
        except Exception as e:
            self.handle_error('_load_yaml', e)

    def handle_error(self, func_name, error):
        print(f"Error in {self.__class__.__name__}.{func_name}: {error}")
        return False

if __name__ == "__main__":
    data_importer = DataImporter()
    url = "uploaded_files/space.csv"
    data_importer.import_data_from_file("test11", url)