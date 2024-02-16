from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import AlbertTokenizer, AlbertForSequenceClassification, Trainer, TrainingArguments
import torch

class ALBERTModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        self.label_encoder = LabelEncoder()

    def _preprocess_data(self, X, y):
        encoded_labels = self.label_encoder.fit_transform(y)
        return X, encoded_labels

    def _tokenize_data(self, X):
        tokenized_inputs = self.tokenizer(X, padding=True, truncation=True, return_tensors="pt")
        return tokenized_inputs

    def train(self, X_train, y_train, X_test, y_test):
        # Preprocess data
        X_train, y_train = self._preprocess_data(X_train, y_train)
        X_test, y_test = self._preprocess_data(X_test, y_test)

        # Tokenize data
        train_inputs = self._tokenize_data(X_train)
        test_inputs = self._tokenize_data(X_test)

        # Define model
        self.model = AlbertForSequenceClassification.from_pretrained('albert-base-v2', num_labels=len(set(y_train)))

        # Training arguments
        training_args = TrainingArguments(
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            evaluation_strategy='epoch',
            logging_dir='./logs',
            output_dir='./results',
            num_train_epochs=3,
            logging_steps=100,
            save_steps=1000,
            warmup_steps=500,
            weight_decay=0.01,
            logging_first_step=True,
            load_best_model_at_end=True,
            metric_for_best_model='accuracy'
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=torch.utils.data.TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], torch.tensor(y_train)),
            eval_dataset=torch.utils.data.TensorDataset(test_inputs['input_ids'], test_inputs['attention_mask'], torch.tensor(y_test)),
            compute_metrics=lambda p: {'accuracy': accuracy_score(p.predictions.argmax(-1), p.label_ids),
                                        'f1_score': f1_score(p.predictions.argmax(-1), p.label_ids, average='weighted')}
        )

        # Train the model
        trainer.train()

        # Save model
        self.model.save_pretrained(self.model_path)

        # Evaluate the model
        evaluation_results = trainer.evaluate(eval_dataset=torch.utils.data.TensorDataset(test_inputs['input_ids'], test_inputs['attention_mask'], torch.tensor(y_test)))

        return evaluation_results['accuracy'], evaluation_results['f1_score']

# Example usage:
if __name__ == "__main__":
    X_train = ["example text 1", "example text 2", "example text 3"]
    y_train = ["label1", "label2", "label3"]
    X_test = ["example text 4", "example text 5"]
    y_test = ["label1", "label2"]

    model_path = "./albert_model"

    albert_model = ALBERTModel(model_path)
    accuracy, f1_score = albert_model.train(X_train, y_train, X_test, y_test)

    print("Accuracy:", accuracy)
    print("F1 Score:", f1_score)
