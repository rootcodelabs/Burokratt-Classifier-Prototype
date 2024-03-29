from sklearn.model_selection import train_test_split
from nlp_models.bert import BERTTrainer
from nlp_models.albert import ALBERTTrainer
from nlp_models.xlnet import XLNetTrainer
from database.database_connection import SQLiteDatabase
import time

class CoreClassifierTrain:
    def __init__(self):
        pass

    def get_dataset(self,class_name_list):
        dataset = {}
        for class_name in class_name_list:
            query = f"""SELECT data, class_name FROM data_info WHERE class_name = '{class_name.upper()}'"""
            result = SQLiteDatabase().execute_query(query)
            for row in result:
                data, class_name = row
                if class_name in dataset:
                    dataset[class_name].append(data)
                else:
                    dataset[class_name] = [data]
        return dataset

    def train_and_evaluate(self, datamodel_name, class_name_list, selected_models=['bert', 'albert', 'xlnet'], datamodel_id = f'{str(int(time.time()))}'):
        try:
            data = self.get_dataset(class_name_list)
            if data:
                X = []
                y = []
                for label, examples in data.items():
                    X.extend(examples)
                    y.extend([label] * len(examples))

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                for model_name in selected_models:
                    model = None
                    if model_name == 'bert':
                        print("in bert")
                        model = BERTTrainer(f"A{datamodel_id}_{model_name}", len(class_name_list))
                    elif model_name == 'albert':
                        print("in albert")
                        print(f"{datamodel_id}_{model_name}")
                        model = ALBERTTrainer(f"A{datamodel_id}_{model_name}", len(class_name_list))
                    elif model_name == 'xlnet':
                        print("in xlnet")
                        model = XLNetTrainer(f"A{datamodel_id}_{model_name}", len(class_name_list))

                    if model:
                        accuracy, f1_score, class_report_dict, label_encoder_dict = model.train(X_train, y_train, X_test, y_test)
                        print(f'Label Encoder dict : \n {label_encoder_dict}')
                        for class_name_str, class_label in label_encoder_dict.items():
                            result = SQLiteDatabase().insert_record('model_class_info', 
                                    {'datamodel_id': f'A{datamodel_id}_{model_name}', 
                                    'class_name': class_name_str, 'class_label': int(class_label), 
                                    'precision': class_report_dict[str(class_label)]['precision'],
                                    'recall': class_report_dict[str(class_label)]['recall'],
                                    'f1_score': class_report_dict[str(class_label)]['f1-score']})
                        print(f"{model_name.capitalize()} Accuracy:", accuracy)
                        print(f"{model_name.capitalize()} F1 Score:", f1_score)
                        print(f"{model_name.capitalize()} class_report:", class_report_dict)
                        SQLiteDatabase().insert_record('model_info', {'datamodel_id': f'A{datamodel_id}_{model_name}', 'datamodel_name':f'{datamodel_name}_{model_name}', 'accuracy': accuracy, 'f1_score': f1_score})

                return True, []
            else:
                return None, data
        except Exception as e:
            return False, f"Error occurred: {str(e)}"
        finally:
            SQLiteDatabase().close_connection()

    def check_dataset_labels(self, dataset_id):
        try:
            query = f"SELECT data_id, data, label FROM dataset_info WHERE dataset_id = '{dataset_id}'"
            dataset_data = SQLiteDatabase().execute_query(query)
            
            missing_labels = []
            data_with_labels = {}
            for data_id, data, label in dataset_data:
                if label == '':
                    missing_labels.append(data_id)
                else:
                    if label in data_with_labels:
                        data_with_labels[label].append(data)
                    else:
                        data_with_labels[label] = [data]
            
            if missing_labels:
                return False, missing_labels
            else:
                return True, data_with_labels
        except Exception as e:
            return None, f"Error occurred: {str(e)}"