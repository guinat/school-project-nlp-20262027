import argparse
import ast
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)
from datasets import Dataset

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
NUM_EPOCHS = 5
WEIGHT_DECAY = 0.01
SEED = 42
PREDICTION_THRESHOLD = 0.5

LABELS = [
    "crime",
    "fantasy",
    "history",
    "horror",
    "psychology",
    "romance",
    "science",
    "sports",
    "thriller",
    "travel",
]

LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}


def get_device_name() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def mps_bf16_supported() -> bool:
    if not torch.backends.mps.is_available():
        return False
    if hasattr(torch.backends.mps, "is_macos_or_newer"):
        return torch.backends.mps.is_macos_or_newer(14, 0)
    if hasattr(torch.mps, "is_bf16_supported"):
        return bool(torch.mps.is_bf16_supported())
    return False


# Seeding
def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available() and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)


# Data loading
def load_split(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if {"summary", "genre"}.issubset(df.columns):
        text_col, label_col = "summary", "genre"
    else:
        raise ValueError(
            f"{path}: expected columns (text,label) or (summary,genre). "
            f"Found: {list(df.columns)}"
        )

    result = df[[text_col, label_col]].copy()
    result.columns = ["text", "label"]

    result["text"] = result["text"].fillna("").astype(str)
    result["label"] = result["label"].apply(parse_labels)

    unknown = sorted(
        {
            label
            for labels in result["label"]
            for label in labels
            if label not in LABEL2ID
        }
    )
    if unknown:
        raise ValueError(
            f"Unknown labels in {path}: {unknown}. "
            f"Expected: {LABELS}"
        )

    result["label_ids"] = result["label"].apply(
        lambda labels: [
            float(index in {LABEL2ID[label] for label in labels})
            for index in range(len(LABELS))
        ]
    )

    return result


def parse_labels(value) -> list[str]:
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
    else:
        parsed = value

    if isinstance(parsed, str):
        parsed = [parsed]
    if not isinstance(parsed, (list, tuple)) or not parsed:
        raise ValueError(f"Invalid genre labels: {value!r}")

    return [str(label).strip().lower() for label in parsed]


# Tokenization
def tokenize_dataset(df: pd.DataFrame, tokenizer) -> Dataset:
    dataset = Dataset.from_pandas(
        df[["text", "label_ids"]],
        preserve_index=False,
    )

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding=False,
            max_length=MAX_LENGTH,
        )

    dataset = dataset.map(
        tokenize_batch,
        batched=True,
        desc="Tokenizing",
    )

    dataset = dataset.rename_column("label_ids", "labels")
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    return dataset


def precision_settings() -> tuple[str, str, dict]:
    device = get_device_name()
    use_fp16 = device == "cuda"
    use_bf16 = device == "mps" and mps_bf16_supported()
    if use_bf16:
        precision = "bf16"
    elif use_fp16:
        precision = "fp16"
    else:
        precision = "fp32"
    return device, precision, {
        "fp16": use_fp16,
        "bf16": use_bf16,
        "dataloader_pin_memory": device == "cuda",
    }


def resolve_model_dir(output_dir: str | Path, model_dir: str | None) -> Path:
    path = Path(model_dir) if model_dir else Path(output_dir) / "best_model"
    has_weights = (path / "model.safetensors").exists() or (
        path / "pytorch_model.bin"
    ).exists()
    if not path.exists() or not has_weights:
        raise FileNotFoundError(
            f"No saved model found at {path}. "
            "Train first with: .venv/bin/python src/transformer.py"
        )
    return path


def load_eval_trainer(model_dir: str | Path, output_dir: str):
    device, _, precision_kwargs = precision_settings()
    print(f"Device: {device}")
    print(f"Loading model from: {model_dir}")
    print("Skipping training.")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_dir,
            per_device_eval_batch_size=BATCH_SIZE,
            report_to="none",
            **precision_kwargs,
        ),
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    return trainer, tokenizer


def _metrics_table(title: str, metrics: dict, mapping: list[tuple[str, str]]) -> str:
    rows = [
        f"| {label} | {float(metrics[key]):.4f} |"
        for label, key in mapping
        if key in metrics
    ]
    if not rows:
        return ""
    return (
        f"## {title}\n\n"
        "| Metric | Value |\n| --- | --- |\n"
        + "\n".join(rows)
        + "\n"
    )


def write_model_card(model_dir: Path, repo_id: str) -> None:
    output_dir = model_dir.parent
    val_mapping = [
        ("F1 macro", "eval_f1_macro"),
        ("F1 micro", "eval_f1_micro"),
        ("Precision macro", "eval_precision_macro"),
        ("Recall macro", "eval_recall_macro"),
        ("Accuracy (exact match)", "eval_accuracy"),
    ]
    test_mapping = [
        ("F1 macro", "f1_macro"),
        ("F1 micro", "f1_micro"),
        ("Precision macro", "precision_macro"),
        ("Recall macro", "recall_macro"),
        ("Accuracy (exact match)", "accuracy"),
    ]

    metrics_block = ""
    config_path = output_dir / "config.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        metrics_block += _metrics_table(
            "Metrics (validation)",
            cfg.get("validation_metrics", {}),
            val_mapping,
        )

    test_path = output_dir / "test_metrics.json"
    if test_path.exists():
        with open(test_path, encoding="utf-8") as f:
            test_metrics = json.load(f)
        if metrics_block:
            metrics_block += "\n"
        metrics_block += _metrics_table(
            "Metrics (test)",
            test_metrics,
            test_mapping,
        )

    labels = ", ".join(f"`{label}`" for label in LABELS)
    card = f"""---
library_name: transformers
pipeline_tag: text-classification
tags:
  - distilbert
  - multi-label
  - book-genre
base_model: distilbert-base-uncased
---

# DistilBERT — book genre multi-label classifier

Fine-tune of [`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased)
on book summaries. A title can have several genres.

**Labels:** {labels}

**Prediction threshold:** {PREDICTION_THRESHOLD}

{metrics_block}
## Usage

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

repo = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForSequenceClassification.from_pretrained(repo)

text = "A young wizard discovers a hidden school of magic."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
with torch.no_grad():
    probs = torch.sigmoid(model(**inputs).logits)[0]

predicted = [
    model.config.id2label[i]
    for i, probability in enumerate(probs.tolist())
    if probability >= {PREDICTION_THRESHOLD}
]
print(predicted)
```
"""
    (model_dir / "README.md").write_text(card, encoding="utf-8")


def push_model_to_hub(
    model_dir: str | Path,
    repo_id: str | None = None,
    private: bool = False,
) -> str:
    from huggingface_hub import HfApi, whoami
    from huggingface_hub.errors import LocalTokenNotFoundError

    model_dir = Path(model_dir)
    try:
        user = whoami()
    except LocalTokenNotFoundError as exc:
        raise SystemExit(
            "Not logged in to Hugging Face.\n"
            "Run: .venv/bin/hf auth login"
        ) from exc

    username = user["name"]
    if repo_id is None:
        repo_id = f"{username}/distilbert-book-genre-multilabel"
    elif "/" not in repo_id:
        repo_id = f"{username}/{repo_id}"

    write_model_card(model_dir, repo_id)
    print(f"Uploading {model_dir} -> {repo_id}")

    api = HfApi()
    api.create_repo(
        repo_id,
        exist_ok=True,
        private=private,
        repo_type="model",
    )
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=["training_args.bin", ".DS_Store"],
        commit_message="Upload DistilBERT multi-label book genre classifier",
    )

    url = f"https://huggingface.co/{repo_id}"
    print(f"Uploaded to {url}")
    return url


# Metrics used during validation
def compute_metrics(eval_prediction):
    logits, labels = eval_prediction
    probabilities = 1 / (1 + np.exp(-logits))
    predictions = (probabilities >= PREDICTION_THRESHOLD).astype(int)
    labels = labels.astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    accuracy = accuracy_score(labels, predictions)

    return {
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "f1_micro": precision_recall_fscore_support(
            labels,
            predictions,
            average="micro",
            zero_division=0,
        )[2],
    }


# Main training procedure
def train_transformer(
    train_path: str,
    validation_path: str,
    output_dir: str = "results/transformer",
):
    seed_everything(SEED)

    device, precision, precision_kwargs = precision_settings()

    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME}")
    print(f"Max length: {MAX_LENGTH}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Seed: {SEED}")
    print(f"Mixed precision: {precision}")

    train_df = load_split(train_path)
    validation_df = load_split(validation_path)

    print(f"\nTrain examples: {len(train_df)}")
    print(f"Validation examples: {len(validation_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = tokenize_dataset(train_df, tokenizer)
    validation_dataset = tokenize_dataset(validation_df, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.config.problem_type = "multi_label_classification"

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_strategy="epoch",
        report_to="none",
        seed=SEED,
        data_seed=SEED,
        **precision_kwargs,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    print("\nTraining...")
    start_train = time.perf_counter()
    trainer.train()
    train_time = time.perf_counter() - start_train

    print(f"\nTraining time: {train_time:.2f} s")

    print("\nValidation...")
    validation_metrics = trainer.evaluate()
    print(json.dumps(validation_metrics, indent=2, default=float))

    # Save the best model and tokenizer.
    best_dir = Path(output_dir) / "best_model"
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    with open(Path(output_dir) / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": MODEL_NAME,
                "max_length": MAX_LENGTH,
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "epochs": NUM_EPOCHS,
                "weight_decay": WEIGHT_DECAY,
                "prediction_threshold": PREDICTION_THRESHOLD,
                "seed": SEED,
                "device": device,
                "mixed_precision": precision,
                "labels": LABELS,
                "train_time_seconds": train_time,
                "validation_metrics": {
                    k: float(v)
                    for k, v in validation_metrics.items()
                    if isinstance(v, (int, float, np.number))
                },
            },
            f,
            indent=2,
        )

    return trainer, tokenizer, train_time


# Optional final test evaluation
def evaluate_test(
    trainer,
    tokenizer,
    test_path: str,
    output_dir: str,
):
    test_df = load_split(test_path)
    test_dataset = tokenize_dataset(test_df, tokenizer)

    print("\nFinal Test Evaluation...")
    start = time.perf_counter()
    predictions = trainer.predict(test_dataset)
    inference_time = time.perf_counter() - start

    y_true = predictions.label_ids.astype(int)
    probabilities = 1 / (1 + np.exp(-predictions.predictions))
    y_pred = (probabilities >= PREDICTION_THRESHOLD).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "f1_micro": precision_recall_fscore_support(
            y_true,
            y_pred,
            average="micro",
            zero_division=0,
        )[2],
        "inference_time_seconds": inference_time,
        "inference_time_per_example_ms": (
            inference_time / len(test_df) * 1000
        ),
    }

    print(json.dumps(metrics, indent=2))

    predictions_df = pd.DataFrame(
        {
            "text": test_df["text"],
            "true_label": [
                str([ID2LABEL[i] for i, value in enumerate(row) if value])
                for row in y_true
            ],
            "predicted_label": [
                str([ID2LABEL[i] for i, value in enumerate(row) if value])
                for row in y_pred
            ],
        }
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    predictions_df.to_csv(
        output_path / "test_predictions.csv",
        index=False,
    )

    with open(output_path / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {k: float(v) for k, v in metrics.items()},
            f,
            indent=2,
        )

    print(f"\nPredictions saved to {output_path / 'test_predictions.csv'}")
    return metrics


# CLI
def main():
    parser = argparse.ArgumentParser(
        description="Train the TextBench DistilBERT classifier."
    )

    parser.add_argument(
        "--train",
        default="data/processed/train.csv",
        help="Path to training CSV.",
    )
    parser.add_argument(
        "--validation",
        default="data/processed/validation.csv",
        help="Path to validation CSV.",
    )
    parser.add_argument(
        "--test",
        default="data/processed/test.csv",
        help="Path to test CSV.",
    )
    parser.add_argument(
        "--output",
        default="results/transformer",
        help="Output directory.",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate a saved model on the test set. Does not train.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Saved model directory. Defaults to {output}/best_model.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload the saved model to the Hugging Face Hub.",
    )
    parser.add_argument(
        "--hub-repo",
        default=None,
        help="Hub repo id (username/name). Defaults to <you>/distilbert-book-genre-multilabel.",
    )
    parser.add_argument(
        "--hub-private",
        action="store_true",
        help="Create the Hub repository as private.",
    )

    args = parser.parse_args()

    if args.evaluate_test or args.push_to_hub:
        model_dir = resolve_model_dir(args.output, args.model)
        if args.evaluate_test:
            trainer, tokenizer = load_eval_trainer(model_dir, args.output)
            evaluate_test(
                trainer,
                tokenizer,
                args.test,
                args.output,
            )
        if args.push_to_hub:
            push_model_to_hub(
                model_dir,
                args.hub_repo,
                private=args.hub_private,
            )
        return

    train_transformer(
        args.train,
        args.validation,
        args.output,
    )


if __name__ == "__main__":
    main()
