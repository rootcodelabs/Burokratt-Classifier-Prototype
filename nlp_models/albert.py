from classification_proccesor import ClassificationReportParser
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from transformers import AlbertTokenizer, AlbertForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.metrics import classification_report

class ALBERTTrainer:
    def __init__(self, datamodel_id):
        try:
            print("!1")
            self.model_path = f'nlp_models/{datamodel_id}/'
            print("!1")
            self.model = None
            print("!1")
            self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
            print("!1")
            self.label_encoder = LabelEncoder()
            print("!1")
        except Exception as e:
            print("Error in albert")
            print(e)

    def _preprocess_data(self, X, y):
        encoded_labels = self.label_encoder.fit_transform(y)
        return X, encoded_labels

    def _tokenize_data(self, X):
        tokenized_inputs = self.tokenizer(X, padding=True, truncation=True, return_tensors="pt")
        return tokenized_inputs

    def train(self, X_train, y_train, X_test, y_test):
        try:
            print("@1")
            X_train, y_train = self._preprocess_data(X_train, y_train)
            print("@1")
            X_test, y_test = self._preprocess_data(X_test, y_test)
            print("@1")

            label_encoder_dict = dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))
            print("@1")

            train_inputs = self._tokenize_data(X_train)
            print("@1")
            test_inputs = self._tokenize_data(X_test)
            print("@1")

            self.model = AlbertForSequenceClassification.from_pretrained('albert-base-v2', num_labels=len(set(y_train)))
            print("@1")

            training_args = TrainingArguments(
                        per_device_train_batch_size=8,
                        per_device_eval_batch_size=8,
                        evaluation_strategy='epoch',
                        logging_dir='./logs',
                        output_dir='./results',
                        num_train_epochs=3,
                        logging_steps=100,
                        save_steps=500,  # Changed to match evaluation strategy
                        warmup_steps=500,
                        weight_decay=0.01,
                        logging_first_step=True,
                        load_best_model_at_end=True,
                        metric_for_best_model='accuracy'
                    )
            print("@1")


            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=torch.utils.data.TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], torch.tensor(y_train)),
                eval_dataset=torch.utils.data.TensorDataset(test_inputs['input_ids'], test_inputs['attention_mask'], torch.tensor(y_test)),
            )

            trainer.train()

            self.model.save_pretrained(self.model_path)

            evaluation_results = trainer.evaluate()

            y_pred = trainer.predict(test_dataset=trainer.eval_dataset)[0]

            class_report = classification_report(y_test, y_pred, target_names=self.label_encoder.classes_)

            parser = ClassificationReportParser(class_report)

            # Parse the report
            class_report_dict = parser.parse_report()

            return evaluation_results['eval_accuracy'], evaluation_results['eval_f1_score'], class_report_dict, label_encoder_dict
        except Exception as e:
            print("Error in albert train")
            print(e)

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
