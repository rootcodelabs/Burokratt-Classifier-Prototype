from nlp_models.bert import BERTClassifier
from nlp_models.albert import ALBERTClassifier
from nlp_models.xlnet import XLNetClassifier

class TextClassifier:
    def __init__(self):
        pass

    def classify_text(self, datamodel_id, text):
        if not datamodel_id:
            raise ValueError("Missing datamodel_id")
        if not text:
            raise ValueError("Missing text")

        self.datamodel_id = datamodel_id
        self.base_model = datamodel_id.split('_')[-1]
        
        try:
            if self.base_model == 'bert':
                classifier = BERTClassifier(self.datamodel_id)
            elif self.base_model == 'albert':
                classifier = ALBERTClassifier(self.datamodel_id)
            elif self.base_model == 'xlnet':
                classifier = XLNetClassifier(self.datamodel_id)
            else:
                raise ValueError("Unsupported base model")
            
            prediction = classifier.classify_text(text)
            return prediction
        
        except FileNotFoundError:
            raise FileNotFoundError(f"Model files not found for datamodel_id: {self.datamodel_id}")
        
        except Exception as e:
            raise RuntimeError(f"Error while classifying text: {str(e)}")
