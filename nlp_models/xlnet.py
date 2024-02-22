import os
import numpy as np
from classification_proccesor import ClassificationReportParser
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from transformers import XLNetTokenizer, XLNetForSequenceClassification, AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from sklearn.metrics import classification_report

class XLNetTrainer:
    def __init__(self, datamodel_id, classes_number):
        print("^0")
        self.model_save_path = f'nlp_models/{datamodel_id}/model'
        print("^0")
        self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        print("^0")
        self.model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=classes_number)
        print("^0")
        self.label_encoder = LabelEncoder()
        print("^0")

    def train(self, X_train, y_train, X_test, y_test):
        print("1. Encoding labels...")
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        label_encoder_dict = dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))

        print("2. Tokenizing inputs...")
        X_train_tokens = self.tokenizer(X_train, padding=True, truncation=True, max_length=256, return_tensors='pt')
        X_test_tokens = self.tokenizer(X_test, padding=True, truncation=True, max_length=256, return_tensors='pt')

        print("3. Creating DataLoaders...")
        train_dataset = TensorDataset(X_train_tokens['input_ids'], X_train_tokens['attention_mask'], torch.tensor(y_train_encoded))
        test_dataset = TensorDataset(X_test_tokens['input_ids'], X_test_tokens['attention_mask'], torch.tensor(y_test_encoded))

        print("^1")
        batch_size = 8
        train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)

        print("^1")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        print("^1")
        optimizer = AdamW(self.model.parameters(), lr=2e-5)
        criterion = torch.nn.CrossEntropyLoss()

        print("^1")
        epochs = 1
        for epoch in range(epochs):
            print("^2")
            self.model.train()
            train_loss = 0.0
            for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{epochs}', unit='batches'):
                print("^3")
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                train_loss += loss.item()  # Accumulate loss here
                loss.backward()
                optimizer.step()

            print("^4")
            test_loss, test_accuracy, test_f1 = self.evaluate(test_dataloader, device, criterion)
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_dataloader)}, Test Loss: {test_loss}, Test Accuracy: {test_accuracy}, Test F1: {test_f1}")

        print("^5")
        self.model.save_pretrained(self.model_save_path)

        predictions, true_labels = self.get_predictions(test_dataloader, device)
        class_report = classification_report(true_labels, predictions)
        parser = ClassificationReportParser(class_report)

        # Parse the report
        class_report_dict = parser.parse_report()
        print("Class report")
        print(class_report_dict)

        return test_accuracy, test_f1, class_report_dict, label_encoder_dict

    def evaluate(self, dataloader, device, criterion):
        self.model.eval()
        eval_loss = 0
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                eval_loss += loss.item()

                logits = outputs.logits.detach().cpu().numpy()
                label_ids = labels.to('cpu').numpy()
                predictions.extend(np.argmax(logits, axis=1))
                true_labels.extend(label_ids)

        eval_loss = eval_loss / len(dataloader)
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted')

        return eval_loss, accuracy, f1

    def get_predictions(self, dataloader, device):
        self.model.eval()
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits.detach().cpu().numpy()
                label_ids = labels.to('cpu').numpy()

                predictions.extend(np.argmax(logits, axis=1))
                true_labels.extend(label_ids)

        return predictions, true_labels

class XLNetClassifier:
    def __init__(self, datamodel_id):
        self.model_path = f'nlp_models/{datamodel_id}/model'
        self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        self.model = XLNetForSequenceClassification.from_pretrained(self.model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Set model to evaluation mode
        self.model.eval()
        # Send model to appropriate device
        self.model.to(self.device)

    def classify_text(self, text):
        # Tokenize input text
        inputs = self.tokenizer(text, padding=True, truncation=True, max_length=256, return_tensors="pt")
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        # Get model prediction
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()

        return predicted_class
