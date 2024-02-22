from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score
import torch
from sklearn.metrics import classification_report
from classification_proccesor import ClassificationReportParser
from sklearn.preprocessing import LabelEncoder

class BERTTrainer:
    def __init__(self, datamodel_id, classes_number):
        try:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=classes_number)
            self.model_path = f'nlp_models/{datamodel_id}/'
        except Exception as e:
            print("Error at BERT Base class")
            print(e)

    def train(self, X_train, y_train_str, X_test, y_test_str):
        try:
            train_encodings = self.tokenizer(X_train, truncation=True, padding=True)
            test_encodings = self.tokenizer(X_test, truncation=True, padding=True)

            y_train = []
            y_test = []
            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(y_train_str)
            y_test = label_encoder.fit_transform(y_test_str)
            label_encoder_dict = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

            train_labels = torch.tensor(y_train)
            test_labels = torch.tensor(y_test)

            train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_encodings['input_ids']),
                                                        torch.tensor(train_encodings['attention_mask']),
                                                        train_labels)
            test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_encodings['input_ids']),
                                                        torch.tensor(test_encodings['attention_mask']),
                                                        test_labels)

            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

            optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
            criterion = torch.nn.CrossEntropyLoss()

            self.model.train()
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

            class_report = classification_report(y_test, y_pred)

            self.model.save_pretrained(self.model_path)

            parser = ClassificationReportParser(class_report)

            # Parse the report
            class_report_dict = parser.parse_report()

            return accuracy, f1, class_report_dict, label_encoder_dict
        except Exception as e:
            print("Error in bert train")
            print(e)

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
