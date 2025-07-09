import argparse
import json
import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from app.utils import set_seed

class TextDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128):
        self.examples = []
        with open(data_path, "r") as f:
            for line in f:
                obj = json.loads(line)
                self.examples.append(obj)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        enc = self.tokenizer(
            item["text"], truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )
        label = 1 if item["label"].lower() == "positive" else 0
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long),
        }

def train(args):
    set_seed(42)
    # load base model/tokenizer
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = TextDataset(args.data, tokenizer)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    model.train()
    for epoch in range(args.epochs):
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            inputs = {
                k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]
            }
            labels = batch["labels"].to(device)
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    # save fine‑tuned weights
    os.makedirs("model", exist_ok=True)
    model.save_pretrained("model")
    tokenizer.save_pretrained("model")
    print("Saved fine‑tuned model to ./model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-data", required=True, help="Path to JSONL data")
    parser.add_argument("-epochs", type=int, default=3)
    parser.add_argument("-lr", type=float, default=3e-5)
    args = parser.parse_args()
    train(args)
