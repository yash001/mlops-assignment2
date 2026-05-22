"""
inference.py — Predict the Goodreads genre of one or more review texts using
the fine-tuned DistilBERT model published on the Hugging Face Hub.

Usage:
    # Single review passed as a CLI argument
    python inference.py "A haunting collection of verses about loss."

    # Pipe one review per line from stdin
    echo "Dragons, sorcery, and an ancient prophecy." | python inference.py

    # Import and call from Python
    from inference import predict
    predict("A page-turning whodunit set in 1920s London.")
"""
import os
import sys
from typing import List, Union

import torch
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
)

HF_USERNAME = os.environ.get("HF_USERNAME", "mehtayash12345678")
HF_REPO = f"{HF_USERNAME}/distilbert-goodreads-genres"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 512

# Cache the model + tokenizer at module load so repeated calls don't reload.
_TOKENIZER = None
_MODEL = None


def _ensure_loaded():
    global _TOKENIZER, _MODEL
    if _TOKENIZER is None or _MODEL is None:
        print(f"Loading {HF_REPO} on {DEVICE} ...", file=sys.stderr)
        _TOKENIZER = DistilBertTokenizerFast.from_pretrained(HF_REPO)
        _MODEL = DistilBertForSequenceClassification.from_pretrained(HF_REPO).to(DEVICE)
        _MODEL.eval()


def predict(text: Union[str, List[str]]) -> Union[str, List[str]]:
    """Return the predicted genre label (or list of labels) for the input."""
    _ensure_loaded()
    single = isinstance(text, str)
    texts = [text] if single else list(text)

    enc = _TOKENIZER(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        logits = _MODEL(**enc).logits
    ids = logits.argmax(-1).tolist()
    labels = [_MODEL.config.id2label[i] for i in ids]
    return labels[0] if single else labels


def _read_input():
    """Read text either from CLI args or stdin (one review per line)."""
    if len(sys.argv) > 1:
        return [" ".join(sys.argv[1:])]
    if not sys.stdin.isatty():
        return [line.strip() for line in sys.stdin if line.strip()]
    # Fallback: a hard-coded example so `python inference.py` always works.
    return ["A sweeping epic of dragons, lost kingdoms, and an unlikely hero."]


def main():
    texts = _read_input()
    preds = predict(texts)
    for txt, lbl in zip(texts, preds):
        preview = txt if len(txt) < 80 else txt[:77] + "..."
        print(f"[{lbl}]  {preview}")


if __name__ == "__main__":
    main()
