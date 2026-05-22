"""
evaluate.py — Reload the fine-tuned model from the Hugging Face Hub and
evaluate it on a fresh balanced test split drawn from the Goodreads dataset.

Prints overall accuracy, weighted F1, and per-class precision/recall/F1
via sklearn.classification_report. Also writes `eval_report.json`.

Usage:
    python evaluate.py
"""
import gzip
import json
import os
import random

import requests
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
)

# ---------------------------------------------------------------------------
# Configuration (must match train.py)
# ---------------------------------------------------------------------------
HF_USERNAME = os.environ.get("HF_USERNAME", "mehtayash12345678")
HF_REPO = f"{HF_USERNAME}/distilbert-goodreads-genres"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 512
EVAL_BATCH_SIZE = 16
PER_GENRE_SAMPLE = 1000
TRAIN_PER_GENRE = 800  # we keep the SAME split logic so the test set lines up
RAW_HEAD = 10000

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


def stream_reviews(url, head=RAW_HEAD, sample_size=2000):
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


def build_test_split():
    test_texts, test_labels = [], []
    for genre, url in GENRE_URLS.items():
        print(f"Loading test reviews for genre: {genre}")
        reviews = stream_reviews(url)
        reviews = random.sample(reviews, PER_GENRE_SAMPLE)
        test_texts.extend(reviews[TRAIN_PER_GENRE:])
        test_labels.extend([genre] * (PER_GENRE_SAMPLE - TRAIN_PER_GENRE))
    return test_texts, test_labels


def batched_predict(model, tokenizer, texts):
    """Predict in mini-batches so we never OOM on the full test set."""
    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), EVAL_BATCH_SIZE):
            batch = texts[i : i + EVAL_BATCH_SIZE]
            enc = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(DEVICE)
            logits = model(**enc).logits
            ids = logits.argmax(-1).tolist()
            preds.extend(model.config.id2label[i] for i in ids)
    return preds


def main():
    random.seed(42)  # same seed as train.py so the test split aligns

    print(f"Loading model from {HF_REPO} ...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(HF_REPO)
    model = DistilBertForSequenceClassification.from_pretrained(HF_REPO).to(DEVICE)

    test_texts, test_labels = build_test_split()
    print(f"Test set size: {len(test_texts)}")

    pred_labels = batched_predict(model, tokenizer, test_texts)

    acc = accuracy_score(test_labels, pred_labels)
    f1 = f1_score(test_labels, pred_labels, average="weighted")
    print(f"\nAccuracy: {acc:.4f}")
    print(f"Weighted F1: {f1:.4f}\n")
    print(classification_report(test_labels, pred_labels))

    report = classification_report(test_labels, pred_labels, output_dict=True)
    report["overall_accuracy"] = acc
    report["overall_f1"] = f1
    with open("eval_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Wrote eval_report.json")


if __name__ == "__main__":
    main()
