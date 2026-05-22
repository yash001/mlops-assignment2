# MLOps Assignment 2 — DistilBERT Goodreads Genre Classifier

**Student:** Yash Mehta

**Roll Number:** G25AIT2133

---

## Overview

This project fine-tunes a **DistilBERT** (`distilbert-base-cased`) model to classify Goodreads book reviews into 8 genres: *poetry, children, comics & graphic, fantasy & paranormal, history & biography, mystery/thriller/crime, romance,* and *young adult*. The dataset comes from the [UCSD Book Graph](https://mengtingwan.github.io/data/goodreads.html). For each genre, 10,000 reviews are streamed and 2,000 are randomly sampled; then 1,000 per genre are used (800 train / 200 test), giving 6,400 training and 1,600 test samples. Training was performed on a **Kaggle Notebook** with GPU acceleration enabled (NVIDIA Tesla T4). Experiment tracking was managed using **Weights & Biases (W&B)** (`report_to="wandb"`), and the trained model was uploaded publicly to **Hugging Face Hub**.

---

## Setup Instructions

The primary training was done in the Kaggle Notebook (linked below). The Python scripts below are extracted from the notebook for standalone use.

```bash
# 1. Clone the repository
git clone https://github.com/<yash001>/mlops-assignment2.git

# 2. Open the project folder
cd mlops-assignment-2

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables for API keys
export WANDB_API_KEY="your_wandb_api_key"
export HF_TOKEN="your_huggingface_token"

# 5. Run training
python train.py

# 6. Run evaluation
python evaluate.py

# 7. Run inference
python inference.py "A thrilling mystery with unexpected twists and a dark ending."
```

---

## Training Platform

Training was performed on **Kaggle Notebook** with the NVIDIA Tesla T4 GPU accelerator.
Kaggle Secrets were used to securely store `WANDB_API_KEY` and `HF_TOKEN` using `kaggle_secrets.UserSecretsClient`.

**Kaggle Notebook:**
<PASTE_PUBLIC_KAGGLE_URL_HERE>

---

## Results

These results are from the notebook's `trainer.evaluate()` output (final evaluation after 3 epochs of fine-tuning):

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 0.6081 |
| F1 Score  | 0.6056 |
| Precision | 0.6058 |
| Recall    | 0.6081 |
| Eval Loss | 2.3824 |

---

## Per-class Performance (DistilBERT)

| Genre                  | Precision | Recall | F1-Score |
| ---------------------- | --------- | ------ | -------- |
| children               | 0.64      | 0.67   | 0.65     |
| comics_graphic         | 0.84      | 0.81   | 0.82     |
| fantasy_paranormal     | 0.42      | 0.47   | 0.44     |
| history_biography      | 0.62      | 0.57   | 0.59     |
| mystery_thriller_crime | 0.57      | 0.62   | 0.60     |
| poetry                 | 0.76      | 0.82   | 0.79     |
| romance                | 0.58      | 0.58   | 0.58     |
| young_adult            | 0.42      | 0.33   | 0.37     |

---

## Links

| Resource | Link |
|----------|------|
| Hugging Face Model | https://huggingface.co/mehtayash12345678/distilbert-goodreads-genres |
| W&B Dashboard | <https://wandb.ai/g25ait2133-indian-institute-technology-jodhpur/huggingface/runs/s7z2mf2y> |
| Kaggle Notebook | <https://www.kaggle.com/code/mehtayash12345678/mlops-assignment2-g25ait2133> |
| GitHub Repository | https://github.com/<yash001>/mlops-assignment2 |

---

## Project Structure

```text
├── train.py
├── evaluate.py
├── inference.py
├── requirements.txt
├── README.md
└── mlops_assignment_2-g25ait2133.ipynb
```

---

## Tools & Libraries

- Python
- PyTorch
- Hugging Face Transformers
- Datasets
- Scikit-learn
- Weights & Biases (W&B)
- Kaggle Notebook
- Hugging Face Hub
