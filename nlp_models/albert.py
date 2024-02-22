from classification_proccesor import ClassificationReportParser
import torch
from transformers import AlbertForSequenceClassification, AlbertTokenizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader
from transformers import AdamW
from sklearn.metrics import classification_report
import numpy as np
import os

class ALBERTTrainer:
    def __init__(self, datamodel_id, classes_number):
        self.model_save_location = f'nlp_models/{datamodel_id}/model'
        self.model = AlbertForSequenceClassification.from_pretrained('albert-base-v2', num_labels=classes_number)
        self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

    def train(self, X_train, y_train, X_test, y_test):
        # Encode labels
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)

        label_encoder_dict = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

        # Tokenize input data
        X_train_tokens = self.tokenizer(X_train, padding=True, truncation=True, return_tensors='pt')
        X_test_tokens = self.tokenizer(X_test, padding=True, truncation=True, return_tensors='pt')

        # Create TensorDatasets
        train_dataset = TensorDataset(X_train_tokens['input_ids'], X_train_tokens['attention_mask'], torch.tensor(y_train_encoded))
        test_dataset = TensorDataset(X_test_tokens['input_ids'], X_test_tokens['attention_mask'], torch.tensor(y_test_encoded))

        # Set batch size
        batch_size = 16

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Define optimizer and loss function
        optimizer = AdamW(self.model.parameters(), lr=2e-5)
        criterion = torch.nn.CrossEntropyLoss()

        # Training loop
        self.model.train()
        for epoch in range(1):  # Adjust number of epochs as needed
            for batch in train_loader:
                input_ids, attention_mask, labels = batch
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

        # Evaluation
        self.model.eval()
        predictions = []
        true_labels = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids, attention_mask, labels = batch
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                _, predicted = torch.max(logits, 1)
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        # Calculate accuracy and F1 score
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted')

        print("Label encoder")
        print(label_encoder_dict)

        if not os.path.exists(os.path.dirname(self.model_save_location)):
            os.makedirs(os.path.dirname(self.model_save_location))  

        # Save the trained model
        torch.save(self.model.state_dict(), self.model_save_location)

        class_report = classification_report(y_test_encoded, predictions)
        parser = ClassificationReportParser(class_report)

        # Parse the report
        class_report_dict = parser.parse_report()
        print("Class report")
        print(class_report_dict)

        return accuracy, f1, class_report_dict, label_encoder_dict

class ALBERTClassifier:
    def __init__(self, datamodel_id):
        self.model_location = f'nlp_models/{datamodel_id}/model'
        self.model = AlbertForSequenceClassification.from_pretrained('albert-base-v2', num_labels=2)
        self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def classify_text(self, text):
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).cpu().numpy()[0]
        return predicted_class
