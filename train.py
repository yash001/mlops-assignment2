"""
train.py — Fine-tune DistilBERT for Goodreads genre classification.

What this script does:
  1. Streams Goodreads review data (8 genres) from the UCSD public mirror.
  2. Tokenizes with DistilBertTokenizerFast (max_length=512).
  3. Fine-tunes `distilbert-base-cased` for 3 epochs.
  4. Logs training and evaluation metrics to Weights & Biases.
  5. Pushes the fine-tuned model + tokenizer to the Hugging Face Hub.

Usage:
    export WANDB_API_KEY=...
    export HF_TOKEN=...
    python train.py
"""
import gzip
import json
import os
import random
import warnings

import requests
import torch
import wandb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "distilbert-base-cased"
MAX_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHED_MODEL_DIR = "distilbert-reviews-genres"

HF_USERNAME = os.environ.get("HF_USERNAME", "mehtayash12345678")
HF_REPO = f"{HF_USERNAME}/distilbert-goodreads-genres"

# 8 genres × 1000 reviews × 0.8 train split = 6400 train, 1600 test
PER_GENRE_SAMPLE = 1000
TRAIN_PER_GENRE = 800
RAW_HEAD = 10000  # how many lines to scan from the gzip stream before sampling

GENRE_URLS = {
    "poetry":                 "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz",
    "children":               "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz",
    "comics_graphic":         "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz",
    "fantasy_paranormal":     "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz",
    "history_biography":      "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz",
    "mystery_thriller_crime": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz",
    "romance":                "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz",
    "young_adult":            "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def stream_reviews(url, head=RAW_HEAD, sample_size=2000):
    """Stream a gzipped JSON-lines file from URL and return a random sample."""
    reviews, count = [], 0
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with gzip.open(resp.raw, "rt", encoding="utf-8") as f:
        for line in f:
            reviews.append(json.loads(line)["review_text"])
            count += 1
            if head and count >= head:
                break
    return random.sample(reviews, min(sample_size, len(reviews)))


def build_splits():
    """Build balanced train/test splits across all 8 genres."""
    train_texts, train_labels, test_texts, test_labels = [], [], [], []
    for genre, url in GENRE_URLS.items():
        print(f"Loading reviews for genre: {genre}")
        reviews = stream_reviews(url)
        reviews = random.sample(reviews, PER_GENRE_SAMPLE)
        train_texts.extend(reviews[:TRAIN_PER_GENRE])
        train_labels.extend([genre] * TRAIN_PER_GENRE)
        test_texts.extend(reviews[TRAIN_PER_GENRE:])
        test_labels.extend([genre] * (PER_GENRE_SAMPLE - TRAIN_PER_GENRE))
    print(f"Train: {len(train_texts)} | Test: {len(test_texts)}")
    return train_texts, train_labels, test_texts, test_labels


class GoodreadsDataset(torch.utils.data.Dataset):
    """Thin wrapper that combines tokenized encodings with integer labels."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # 1. Authenticate W&B and Hugging Face -----------------------------------
    wandb_key = os.environ.get("WANDB_API_KEY")
    hf_token = os.environ.get("HF_TOKEN")
    if not wandb_key or not hf_token:
        raise RuntimeError(
            "WANDB_API_KEY and HF_TOKEN must be set as environment variables. "
            "On Kaggle, register them under Add-ons → Secrets and load with "
            "kaggle_secrets.UserSecretsClient()."
        )
    # `verify=False` skips the post-login viewer query that occasionally
    # crashes with `TypeError: the JSON object must be str, bytes or
    # bytearray, not NoneType` on certain wandb versions.
    wandb.login(key=wandb_key, verify=False)

    # 2. Build data ----------------------------------------------------------
    train_texts, train_labels, test_texts, test_labels = build_splits()

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    unique_labels = sorted(set(train_labels))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}

    train_enc = tokenizer(
        train_texts, truncation=True, padding=True, max_length=MAX_LENGTH
    )
    test_enc = tokenizer(
        test_texts, truncation=True, padding=True, max_length=MAX_LENGTH
    )
    train_ds = GoodreadsDataset(train_enc, [label2id[y] for y in train_labels])
    test_ds = GoodreadsDataset(test_enc, [label2id[y] for y in test_labels])

    # 3. Model ---------------------------------------------------------------
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    ).to(DEVICE)

    # 4. Training arguments --------------------------------------------------
    args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=10,
        per_device_eval_batch_size=16,
        learning_rate=5e-5,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        eval_strategy="steps",
        report_to="wandb",
        run_name="distilbert-run-1",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    # 5. Train, evaluate, save ----------------------------------------------
    trainer.train()
    trainer.save_model(CACHED_MODEL_DIR)
    eval_results = trainer.evaluate()
    print("Final evaluation:", eval_results)

    wandb.log(
        {
            "final/loss":     eval_results.get("eval_loss", 0),
            "final/accuracy": eval_results.get("eval_accuracy", 0),
            "final/f1":       eval_results.get("eval_f1", 0),
        }
    )

    # 6. Push to Hugging Face -----------------------------------------------
    from huggingface_hub import login as hf_login
    hf_login(token=hf_token)
    model.push_to_hub(HF_REPO)
    tokenizer.push_to_hub(HF_REPO)
    wandb.run.summary["huggingface_model"] = f"https://huggingface.co/{HF_REPO}"
    wandb.finish()

    print(f"Done. Model published at https://huggingface.co/{HF_REPO}")


if __name__ == "__main__":
    random.seed(42)
    main()
