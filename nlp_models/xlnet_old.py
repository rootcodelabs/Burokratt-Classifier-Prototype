import torch
from transformers import XLNetTokenizer, XLNetForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report

class CustomDataset(Dataset):
    def __init__(self, X, y, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.text = X
        self.targets = y
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = str(self.text[idx])
        target = self.targets[idx]

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True
        )

        return {
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(target, dtype=torch.long)
        }

class XLNetTrainer:
    def __init__(self, datamodel_id):
        self.model_path = f'nlp_models/{datamodel_id}/'
        self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        self.model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=1)

    #Epoch here
    def train(self, train_loader, test_loader, num_epochs=1, learning_rate=2e-5):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0

            for batch in tqdm(train_loader, desc="Epoch " + str(epoch)):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                train_loss += loss.item()

                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch} - Training loss: {train_loss / len(train_loader)}")

        self.model.save_pretrained(self.model_path)

        self.model.eval()
        predictions, true_labels = [], []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                preds = torch.argmax(logits, dim=1).flatten()
                predictions.extend(preds.cpu().detach().numpy())
                true_labels.extend(labels.cpu().detach().numpy())

        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted')

        print(f"Accuracy: {accuracy}, F1 Score: {f1}")

        class_report = classification_report(true_labels, predictions)
        print(class_report)

        return accuracy, f1, class_report

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

# # Example usage:
# if __name__ == "__main__":
#     X_train = [...]  # List of training texts
#     y_train = [...]  # List of corresponding training labels
#     X_test = [...]   # List of test texts
#     y_test = [...]   # List of corresponding test labels

#     train_dataset = CustomDataset(X_train, y_train, XLNetTokenizer.from_pretrained('xlnet-base-cased'), max_len=128)
#     test_dataset = CustomDataset(X_test, y_test, XLNetTokenizer.from_pretrained('xlnet-base-cased'), max_len=128)

#     train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

#     model_path = "path_to_save_model"

#     xlnet_trainer = XLNetTrainer(model_path)
#     accuracy, f1_score = xlnet_trainer.train(train_loader, test_loader)

#     # Example of using the classify_text method
#     text_to_classify = "Your input text here"
#     xlnet_classifier = XLNetClassifier(model_path)
#     prediction = xlnet_classifier.classify_text(text_to_classify)
#     print("Prediction:", prediction)
