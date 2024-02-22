import torch
from transformers import AlbertForSequenceClassification, AlbertTokenizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader
from transformers import AdamW
import numpy as np

class ALBERTTrainer:
    def __init__(self, model_save_location):
        self.model_save_location = model_save_location
        self.model = AlbertForSequenceClassification.from_pretrained('albert-base-v2', num_labels=2)
        self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

    def train(self, X_train, y_train, X_test, y_test):
        # Encode labels
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)

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
        for epoch in range(3):  # Adjust number of epochs as needed
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

        # Save the trained model
        torch.save(self.model.state_dict(), self.model_save_location)

        return accuracy, f1

# Example usage:
if __name__ == "__main__":
    X_train = ["text1", "text2", "text3"]
    y_train = ["label1", "label2", "label3"]
    X_test = ["text4", "text5", "text6"]
    y_test = ["label4", "label5", "label6"]

    model_save_location = "nlp_models/model.pth"
    albert_trainer = ALBERTTrainer(model_save_location)
    accuracy, f1 = albert_trainer.train(X_train, y_train, X_test, y_test)
    print(f"Accuracy: {accuracy}, F1 Score: {f1}")

class ALBERTClassifier:
    def __init__(self, datamodel_id):
        self.model_path = f'nlp_models/{datamodel_id}/'
        self.model = None
        self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        self.label_encoder = LabelEncoder()

    def _tokenize_data(self, X):
        tokenized_inputs = self.tokenizer(X, padding=True, truncation=True, return_tensors="pt")
        return tokenized_inputs

    def load_model(self):
        if self.model is None:
            self.model = AlbertForSequenceClassification.from_pretrained(self.model_path)

    def classify_text(self, text):
        self.load_model()

        inputs = self._tokenize_data([text])

        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_class_idx = outputs.logits.argmax().item()

        predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]
        return predicted_class