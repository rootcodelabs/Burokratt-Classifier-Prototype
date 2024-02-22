
from database.database_connection import SQLiteDatabase
SQLiteDatabase().execute_query("DELETE FROM model_info")
SQLiteDatabase().execute_query("DELETE FROM model_class_info")

# result = SQLiteDatabase().insert_record('model_class_info', 
#                                     {'datamodel_id': f'{"1708520823_5_bert"}', 
#                                     'class_name': "ARTS", 'class_label': int(0), 
#                                     'precision': 0.438,
#                                     'recall': 0.8553,
#                                     'f1_score': 0.9541})

# result = SQLiteDatabase().insert_record('model_class_info', 
#                                     {'datamodel_id': f'{"1708520823_5_bert"}', 
#                                     'class_name': "BUSINESS", 'class_label': int(1), 
#                                     'precision': 0.438,
#                                     'recall': 0.8553,
#                                     'f1_score': 0.9541})

# SQLiteDatabase().insert_record('model_info', {'datamodel_id': f'{"1708520823_5_bert"}','datamodel_name':'Alpha', 'accuracy': 0.8, 'f1_score': 0.9})


# result = SQLiteDatabase().insert_record('model_class_info', 
#                                     {'datamodel_id': f'{"1708520823_6_albert"}', 
#                                     'class_name': "SPORTS", 'class_label': int(0), 
#                                     'precision': 0.438,
#                                     'recall': 0.8553,
#                                     'f1_score': 0.9541})

# result = SQLiteDatabase().insert_record('model_class_info', 
#                                     {'datamodel_id': f'{"1708520823_6_albert"}', 
#                                     'class_name': "ARTS", 'class_label': int(1), 
#                                     'precision': 0.438,
#                                     'recall': 0.8553,
#                                     'f1_score': 0.9541})

# SQLiteDatabase().insert_record('model_info', {'datamodel_id': f'{"1708520823_6_albert"}','datamodel_name':'Beta', 'accuracy': 0.8, 'f1_score': 0.9})