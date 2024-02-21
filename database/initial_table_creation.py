from database.database_connection import SQLiteDatabase

db = SQLiteDatabase()

db.create_table('model_info', {'datamodel_id': 'TEXT PRIMARY KEY', 'accuracy': 'REAL', 'f1_score': 'REAL'})

db.create_table_with_composite_key('model_class_info', {'datamodel_id': 'TEXT', 'class_name': 'TEXT', 'class_label': 'INTEGER', 'precision': 'REAL', 'recall': 'REAL', 'f1_score': 'REAL'}, 'datamodel_id', 'class_name')

db.create_table('dataset_info', {'dataset_id': 'TEXT PRIMARY KEY', 'dataset_name': 'TEXT'})

db.create_table_with_composite_key('data_info', {'dataset_id': 'TEXT', 'data_id': 'TEXT', 'data': 'TEXT', 'class_name': 'TEXT'}, 'dataset_id', 'data_id')

db.create_table('class_info', {'class_name': 'TEXT PRIMARY KEY'})

db.create_table_with_composite_key('class_dataset_info', {'class_name': 'TEXT', 'dataset_id': 'TEXT'}, 'class_name', 'dataset_id')

db.conn.commit()

db.view_table_structure()

db.close_connection()
