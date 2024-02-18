import csv
import json
import yaml
from time import time
from database.database_connection import SQLiteDatabase

class DataImporter:
    def __init__(self):
        self.db = SQLiteDatabase()

    def import_data_from_file(self, file_location):
        try:
            data, file_type = self._load_data(file_location)
            print("####")
            print(data)
            print("####")
            if data:
                dataset_id = int(time())
                print(file_type)
                print(type(data))
                if file_type == "json" and type(data)==dict:
                    print("in")
                    data_point_count = 0
                    for label, values in data.items():
                        for string_value in values:
                            record = {
                                'dataset_id': str(dataset_id),
                                'data_id': str(data_point_count),
                                'data': str(string_value),
                                'label': str(label)
                            }
                            data_point_count += 1
                            print(record)
                            print("=====")
                            self.db.insert_record('dataset_info', record)
                else:
                    for idx, datum in enumerate(data):
                        print(datum)
                        record = {
                            'dataset_id': str(dataset_id),
                            'data_id': str(idx),
                            'data': str(datum),
                            'label': ''
                        }
                        print(record)
                        print("=====")
                        self.db.insert_record('dataset_info', record)
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
            data = []
            with open(file_location, 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    data.extend(row)
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

# # Example Usage:
# if __name__ == "__main__":
#     importer = DataImporter()
#     file_location = 'test.json'
#     importer.import_data_from_file(file_location)
