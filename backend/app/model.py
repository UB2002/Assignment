import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
PRETRAINED_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

class SentimentModel:
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._load()

    def _load(self):
        # Check if local model exists
        if os.path.isdir(MODEL_DIR) and os.listdir(MODEL_DIR):
            path = MODEL_DIR
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForSequenceClassification.from_pretrained(path)
        else:
            # Load from pretrained, then save to MODEL_DIR
            self.tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
            self.model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_MODEL)
            os.makedirs(MODEL_DIR, exist_ok=True)
            self.model.save_pretrained(MODEL_DIR)
            self.tokenizer.save_pretrained(MODEL_DIR)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
            scores = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        label_idx = int(scores.argmax())
        label = self.model.config.id2label[label_idx]
        return {"label": label.lower(), "score": float(scores[label_idx])}


if __name__ =="__main__":
    s = SentimentModel()
    text = "this is a good product"
    ans = s.predict(text)
    print(ans)