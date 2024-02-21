from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score
import torch
from sklearn.metrics import classification_report
from numberical_embed import StringConverter

class BERTTrainer:
    def __init__(self, datamodel_id):
        try:
            self.converter = StringConverter()
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
            self.model_path = f'nlp_models/{datamodel_id}/'
        except Exception as e:
            print("Error at BERT Base class")
            print(e)
            print("#####")

    def train(self, X_train, y_train_str, X_test, y_test_str):
        try:
            print("!1")
            train_encodings = self.tokenizer(X_train, truncation=True, padding=True)
            print("!1")
            test_encodings = self.tokenizer(X_test, truncation=True, padding=True)
            print("!1")

            y_train = []
            y_test = []
            for example in y_train:
                y_train.append(self.converter.string_to_integer(example))

            for example in y_test:
                y_test.append(self.converter.string_to_integer(example))

            train_labels = torch.tensor(y_train)
            print("!1")
            test_labels = torch.tensor(y_test)
            print("!1")

            train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_encodings['input_ids']),
                                                        torch.tensor(train_encodings['attention_mask']),
                                                        train_labels)
            print("!1")
            test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_encodings['input_ids']),
                                                        torch.tensor(test_encodings['attention_mask']),
                                                        test_labels)
            print("!1")

            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
            print("!1")
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)
            print("!1")

            optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
            print("!1")
            criterion = torch.nn.CrossEntropyLoss()
            print("!1")

            self.model.train()
            print("!1")
            for epoch in range(1):  # 1 epoch
                for batch in train_loader:
                    input_ids, attention_mask, labels = batch
                    optimizer.zero_grad()
                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()

            self.model.eval()
            y_pred = []
            with torch.no_grad():
                for batch in test_loader:
                    input_ids, attention_mask, labels = batch
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    _, predicted = torch.max(logits, 1)
                    y_pred.extend(predicted.tolist())

            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')

            # Compute classification report
            class_report = classification_report(y_test, y_pred)

            self.model.save_pretrained(self.model_path)

            return accuracy, f1, class_report
        except Exception as e:
            print("Error in bert train")
            print(e)
            print("############")

class BERTClassifier:
    def __init__(self, datamodel_id):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model_path = f'nlp_models/{datamodel_id}/'

    def classify_text(self, text):
        loaded_model = BertForSequenceClassification.from_pretrained(self.model_path)

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            outputs = loaded_model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            _, predicted_class = torch.max(probabilities, dim=1)

        return predicted_class.item()
