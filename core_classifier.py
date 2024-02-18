from sklearn.model_selection import train_test_split
from nlp_models.bert import BERTTrainer
from nlp_models.albert import ALBERTTrainer
from nlp_models.xlnet import XLNetTrainer
from database.database_connection import SQLiteDatabase
import json
import time

class CoreClassifierTrain:
    def __init__(self):
        pass

    def train_and_evaluate(self, dataset_id, model_id, selected_models=['bert', 'albert', 'xlnet']):
        db = SQLiteDatabase()

        try:
            is_labeled, data = self.check_dataset_labels(db, dataset_id)
            if is_labeled is False:
                return False, data
            elif is_labeled is True:
                X = []
                y = []
                for label, examples in data.items():
                    X.extend(examples)
                    y.extend([label] * len(examples))

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                for model_name in selected_models:
                    model = None
                    if model_name == 'bert':
                        model = BERTTrainer(f"{model_id}_{model_name}")
                    elif model_name == 'albert':
                        model = ALBERTTrainer(f"{model_id}_{model_name}")
                    elif model_name == 'xlnet':
                        model = XLNetTrainer(f"{model_id}_{model_name}")

                    if model:
                        accuracy, f1_score = model.train(X_train, y_train, X_test, y_test)
                        print(f"{model_name.capitalize()} Accuracy:", accuracy)
                        print(f"{model_name.capitalize()} F1 Score:", f1_score)
                        db.insert_record('model_info', {'model_id': f'{model_id}_{model_name}', 'accuracy': accuracy, 'f1_score': f1_score, 'dataset_id': dataset_id})

                return True, []
            else:
                return None, data
        except Exception as e:
            return False, f"Error occurred: {str(e)}"
        finally:
            db.close_connection()

    def check_dataset_labels(self, db, dataset_id):
        try:
            query = f"SELECT data_id, data, label FROM dataset_info WHERE dataset_id = '{dataset_id}'"
            dataset_data = db.execute_query(query)
            
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

# # Example usage:
# if __name__ == "__main__":
#     dataset_id = 'your_dataset_id_here'
#     result = check_dataset_labels(dataset_id)
#     print(result)


# if __name__ == "__main__":
#     json_file_path = 'data.json'

#     with open(json_file_path, 'r') as f:
#         data = json.load(f)

#     # Sample model location
#     model_id = f'{str(int(time.time()))}'

#     # Create an instance of CoreClassifierTrain
#     classifier = CoreClassifierTrain()

#     # Train and evaluate the classifier
#     classifier.train_and_evaluate(data, model_id)
