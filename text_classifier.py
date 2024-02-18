from nlp_models.bert import BERTClassifier
from nlp_models.albert import ALBERTClassifier
from nlp_models.xlnet import XLNetClassifier

class TextClassifier:
    def __init__(self):
        pass

    def classify_text(self, model_id, text):
        if not model_id:
            raise ValueError("Missing model_id")
        if not text:
            raise ValueError("Missing text")

        self.model_id = model_id
        self.base_model = model_id.split('_')[-1]
        
        try:
            if self.base_model == 'bert':
                classifier = BERTClassifier(self.model_id)
            elif self.base_model == 'albert':
                classifier = ALBERTClassifier(self.model_id)
            elif self.base_model == 'xlnet':
                classifier = XLNetClassifier(self.model_id)
            else:
                raise ValueError("Unsupported base model")
            
            prediction = classifier.classify_text(text)
            return prediction
        
        except FileNotFoundError:
            raise FileNotFoundError(f"Model files not found for model_id: {self.model_id}")
        
        except Exception as e:
            raise RuntimeError(f"Error while classifying text: {str(e)}")
