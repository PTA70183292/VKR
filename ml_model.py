from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from config import settings

class SentimentModel:
     def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            settings.model_name,
            load_in_8bit=True,
            device_map="auto"
        )
        self.clf = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            return_all_scores=False
        )
    
     def predict(self, text: str) -> dict:
        result = self.clf(text)[0]
        return {
            "label": result["label"],
            "score": float(result["score"])
        }

sentiment_model = None

def get_sentiment_model() -> SentimentModel:
    global sentiment_model
    if sentiment_model is None:
        sentiment_model = SentimentModel()
    return sentiment_model
