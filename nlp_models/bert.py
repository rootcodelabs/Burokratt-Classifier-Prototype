from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score
import torch

class BERTModel:
    def __init__(self, model_path):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
        self.model_path = model_path

    def train(self, X_train, y_train, X_test, y_test):
        # Tokenize input data
        train_encodings = self.tokenizer(X_train, truncation=True, padding=True)
        test_encodings = self.tokenizer(X_test, truncation=True, padding=True)

        # Convert labels to tensors
        train_labels = torch.tensor(y_train)
        test_labels = torch.tensor(y_test)

        # Create PyTorch datasets
        train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_encodings['input_ids']),
                                                       torch.tensor(train_encodings['attention_mask']),
                                                       train_labels)
        test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_encodings['input_ids']),
                                                      torch.tensor(test_encodings['attention_mask']),
                                                      test_labels)

        # Initialize dataloaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

        # Set up optimizer and loss function
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        criterion = torch.nn.CrossEntropyLoss()

        # Training loop
        self.model.train()
        for epoch in range(3):  # example, you can adjust epochs
            for batch in train_loader:
                input_ids, attention_mask, labels = batch
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

        # Evaluation
        self.model.eval()
        y_pred = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids, attention_mask, labels = batch
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                _, predicted = torch.max(logits, 1)
                y_pred.extend(predicted.tolist())

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Save the model
        self.model.save_pretrained(self.model_path)

        return accuracy, f1
