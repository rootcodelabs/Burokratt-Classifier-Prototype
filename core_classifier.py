from sklearn.model_selection import train_test_split
from nlp_models.bert import BERTModel
from nlp_models.albert import ALBERTModel
from nlp_models.xlnet import XLNetModel
from database.database_connection import SQLiteDatabase
import json
import time

class CoreClassifier:
    def __init__(self):
        pass

    def train_and_evaluate(self, data, model_location):
        db = SQLiteDatabase()
        # Splitting data into features and labels
        X = []
        y = []
        for label, examples in data.items():
            X.extend(examples)
            y.extend([label] * len(examples))

        # Splitting data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize
        model = BERTModel(f"{model_location}bert")
        model = ALBERTModel(f"{model_location}albert")
        model = XLNetModel(f"{model_location}xlnet")        

        # Train the model
        accuracy, f1_score = model.train(X_train, y_train, X_test, y_test)
        # Print accuracy and f1_score
        print("BERT Accuracy:", accuracy)
        print("BERT F1 Score:", f1_score)
        db.insert_record('models', {'model_id': f'{str(int(time.time()))}_bert', 'accuracy': accuracy, 'f1_score': f1_score})
########
        # Train the model
        accuracy, f1_score = model.train(X_train, y_train, X_test, y_test)
        # Print accuracy and f1_score
        print("ALBERTModel Accuracy:", accuracy)
        print("ALBERTModel F1 Score:", f1_score)
        db.insert_record('models', {'model_id': f'{str(int(time.time()))}_albert', 'accuracy': accuracy, 'f1_score': f1_score})
########
        # Train the model
        accuracy, f1_score = model.train(X_train, y_train, X_test, y_test)
        # Print accuracy and f1_score
        print("XLNet Accuracy:", accuracy)
        print("XLNet F1 Score:", f1_score)
        db.insert_record('models', {'model_id': f'{str(int(time.time()))}_xlnet', 'accuracy': accuracy, 'f1_score': f1_score})

# Example usage:
if __name__ == "__main__":
    json_file_path = 'data.json'

    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Sample model location
    model_location = f'nlp_models/{str(int(time.time()))}/'

    # Create an instance of CoreClassifier
    classifier = CoreClassifier()

    # Train and evaluate the classifier
    classifier.train_and_evaluate(data, model_location)
