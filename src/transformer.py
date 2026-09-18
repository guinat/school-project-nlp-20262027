import argparse
import ast
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
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

# Seeding 
def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Model: {MODEL_NAME}")
    print(f"Max length: {MAX_LENGTH}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Seed: {SEED}")

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
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
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
        help="Run the final test evaluation. Do not use during tuning.",
    )

    args = parser.parse_args()

    trainer, tokenizer, _ = train_transformer(
        args.train,
        args.validation,
        args.output,
    )

    if args.evaluate_test:
        evaluate_test(
            trainer,
            tokenizer,
            args.test,
            args.output,
        )


if __name__ == "__main__":
    main()
