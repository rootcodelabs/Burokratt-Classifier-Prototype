import os
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from transformers import XLNetTokenizer, XLNetForSequenceClassification, AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

class XLNetTrainer:
    def __init__(self, datamodel_id):
        self.model_save_path = f'nlp_models/{datamodel_id}/model'
        self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        self.model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=2)
        self.label_encoder = LabelEncoder()

    def train(self, X_train, y_train, X_test, y_test):
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)

        # Tokenize inputs
        X_train_tokens = self.tokenizer(X_train, padding=True, truncation=True, max_length=256, return_tensors='pt')
        X_test_tokens = self.tokenizer(X_test, padding=True, truncation=True, max_length=256, return_tensors='pt')

        # Create DataLoader
        train_dataset = TensorDataset(X_train_tokens['input_ids'], X_train_tokens['attention_mask'], torch.tensor(y_train_encoded))
        test_dataset = TensorDataset(X_test_tokens['input_ids'], X_test_tokens['attention_mask'], torch.tensor(y_test_encoded))

        batch_size = 8
        train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)

        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        # Set optimizer and loss function
        optimizer = AdamW(self.model.parameters(), lr=2e-5)
        criterion = torch.nn.CrossEntropyLoss()

        # Training loop
        epochs = 1
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{epochs}', unit='batches'):
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                train_loss += loss.item()

                loss.backward()
                optimizer.step()

            # Evaluate on test set
            test_loss, test_accuracy, test_f1 = self.evaluate(test_dataloader, device, criterion)
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_dataloader)}, Test Loss: {test_loss}, Test Accuracy: {test_accuracy}, Test F1: {test_f1}")

        # Save model
        self.model.save_pretrained(self.model_save_path)

        return test_accuracy, test_f1

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
                logits = outputs.logits
                loss = criterion(logits.view(-1, self.model.config.num_labels), labels)
                eval_loss += loss.item()

                logits = logits.detach().cpu().numpy()
                label_ids = labels.to('cpu').numpy()
                predictions.extend(np.argmax(logits, axis=1))
                true_labels.extend(label_ids)

        eval_loss = eval_loss / len(dataloader)
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted')

        return eval_loss, accuracy, f1

class XLNetClassifier:
    def __init__(self, model_path):
        self.model_path = model_path
        self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

    def classify_text(self, text):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = XLNetForSequenceClassification.from_pretrained(self.model_path)
        model.to(device)
        model.eval()

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            prediction = torch.argmax(logits, dim=1).item()

        return prediction
