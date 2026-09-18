import argparse
import ast
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset

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

DEFAULT_CONFIG = {
    "seed": 42,
    "max_features": 20_000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "hidden_dim": 256,
    "dropout": 0.3,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 64,
    "epochs": 20,
    "patience": 4,
    # Always keep the top label. Add a 2nd/3rd only if its probability is clearly high.
    "extra_label_threshold": 0.8,
}


def get_device_name() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available() and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)


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


def load_split(data) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = pd.read_csv(data)

    if {"summary", "genre"}.issubset(df.columns):
        text_col, label_col = "summary", "genre"
    elif {"text", "label"}.issubset(df.columns):
        text_col, label_col = "text", "label"
    else:
        raise ValueError(
            "Expected columns (summary, genre) or (text, label). "
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
        raise ValueError(f"Unknown labels: {unknown}. Expected: {LABELS}")

    result["label_ids"] = result["label"].apply(
        lambda labels: [
            float(index in {LABEL2ID[label] for label in labels})
            for index in range(len(LABELS))
        ]
    )
    return result


class SparseTfidfDataset(Dataset):
    def __init__(self, matrix, labels: np.ndarray):
        self.matrix = matrix
        self.labels = labels.astype(np.float32)

    def __len__(self) -> int:
        return self.matrix.shape[0]

    def __getitem__(self, index: int):
        row = torch.from_numpy(self.matrix[index].toarray().ravel()).float()
        target = torch.from_numpy(self.labels[index])
        return row, target


class TfidfMLP(nn.Module):
    """TF-IDF bag-of-ngrams -> hidden ReLU layer -> multi-label logits."""

    def __init__(self, input_dim: int, hidden_dim: int, num_labels: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


@dataclass
class DeepLearningModel:
    network: TfidfMLP
    vectorizer: TfidfVectorizer
    config: dict
    device: str
    labels: list[str]
    history: dict


def decode_probabilities(
    probabilities: np.ndarray,
    extra_threshold: float,
) -> np.ndarray:
    """Predict at least one label (argmax). Extra labels only if probability is high."""
    predictions = np.zeros_like(probabilities, dtype=int)
    top = probabilities.argmax(axis=1)
    rows = np.arange(len(probabilities))
    predictions[rows, top] = 1
    extra = probabilities >= extra_threshold
    extra[rows, top] = False
    predictions[extra] = 1
    return predictions


def compute_metrics(
    y_true: np.ndarray,
    logits: np.ndarray,
    extra_threshold: float,
) -> dict:
    probabilities = 1 / (1 + np.exp(-logits))
    predictions = decode_probabilities(probabilities, extra_threshold)
    y_true = y_true.astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        predictions,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "f1_micro": precision_recall_fscore_support(
            y_true,
            predictions,
            average="micro",
            zero_division=0,
        )[2],
    }


def _run_epoch(network, loader, criterion, device, optimizer=None):
    train_mode = optimizer is not None
    network.train(train_mode)

    total_loss = 0.0
    n_examples = 0
    logits_all = []
    labels_all = []

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        if train_mode:
            optimizer.zero_grad()

        logits = network(features)
        loss = criterion(logits, labels)

        if train_mode:
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        n_examples += batch_size
        logits_all.append(logits.detach().cpu().numpy())
        labels_all.append(labels.detach().cpu().numpy())

    return (
        total_loss / max(n_examples, 1),
        np.concatenate(labels_all),
        np.concatenate(logits_all),
    )


def train(train_data, validation_data, config: dict | None = None) -> DeepLearningModel:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    seed_everything(int(cfg["seed"]))
    device = get_device_name()

    train_df = load_split(train_data)
    validation_df = load_split(validation_data)

    vectorizer = TfidfVectorizer(
        max_features=int(cfg["max_features"]),
        ngram_range=tuple(cfg["ngram_range"]),
        min_df=int(cfg["min_df"]),
        lowercase=True,
        stop_words="english",
    )
    x_train = vectorizer.fit_transform(train_df["text"])
    x_validation = vectorizer.transform(validation_df["text"])
    y_train = np.vstack(train_df["label_ids"].to_numpy())
    y_validation = np.vstack(validation_df["label_ids"].to_numpy())

    train_loader = DataLoader(
        SparseTfidfDataset(x_train, y_train),
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
    )
    validation_loader = DataLoader(
        SparseTfidfDataset(x_validation, y_validation),
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
    )

    network = TfidfMLP(
        input_dim=x_train.shape[1],
        hidden_dim=int(cfg["hidden_dim"]),
        num_labels=len(LABELS),
        dropout=float(cfg["dropout"]),
    ).to(device)

    positives = np.clip(y_train.sum(axis=0), 1.0, None)
    negatives = len(y_train) - positives
    pos_weight = torch.tensor(negatives / positives, dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    history = {
        "train_loss": [],
        "validation_loss": [],
        "validation_metrics": [],
    }
    best_f1 = -1.0
    best_state = None
    epochs_without_improve = 0
    extra_threshold = float(cfg["extra_label_threshold"])

    print(f"Device: {device}")
    print(f"Representation: TF-IDF ngrams={cfg['ngram_range']} max_features={cfg['max_features']}")
    print(f"Architecture: Linear({x_train.shape[1]}) -> ReLU -> Dropout({cfg['dropout']}) -> Linear({len(LABELS)})")
    print(f"Loss: BCEWithLogitsLoss with pos_weight (class imbalance)")
    print(f"Optimizer: Adam lr={cfg['learning_rate']} weight_decay={cfg['weight_decay']}")
    print(f"Batch size: {cfg['batch_size']}  Epochs: {cfg['epochs']}  Seed: {cfg['seed']}")
    print(f"Decode: argmax + extra labels only if p >= {extra_threshold}")
    print(f"Train examples: {len(train_df)}  Validation examples: {len(validation_df)}")
    print(f"Vocabulary size: {x_train.shape[1]}")

    print("\nTraining...")
    start_train = time.perf_counter()

    for epoch in range(1, int(cfg["epochs"]) + 1):
        train_loss, _, _ = _run_epoch(
            network, train_loader, criterion, device, optimizer
        )
        with torch.no_grad():
            val_loss, y_true, logits = _run_epoch(
                network, validation_loader, criterion, device
            )
        val_metrics = compute_metrics(y_true, logits, extra_threshold)

        history["train_loss"].append(train_loss)
        history["validation_loss"].append(val_loss)
        history["validation_metrics"].append(val_metrics)

        print(
            f"epoch {epoch:02d}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_f1_macro={val_metrics['f1_macro']:.3f}  "
            f"val_acc={val_metrics['accuracy']:.3f}"
        )

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            best_state = {k: v.detach().cpu().clone() for k, v in network.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= int(cfg["patience"]):
                print(f"Early stopping at epoch {epoch} (best val F1 macro={best_f1:.3f})")
                break

    train_time = time.perf_counter() - start_train
    history["train_time_seconds"] = train_time
    history["best_val_f1_macro"] = best_f1
    print(f"\nTraining time: {train_time:.2f} s")

    if best_state is not None:
        network.load_state_dict(best_state)
    network.to(device)
    network.eval()

    return DeepLearningModel(
        network=network,
        vectorizer=vectorizer,
        config={**cfg, "input_dim": int(x_train.shape[1]), "device": device},
        device=device,
        labels=list(LABELS),
        history=history,
    )


def predict(model: DeepLearningModel, texts) -> list[list[str]]:
    if isinstance(texts, pd.Series):
        texts = texts.tolist()
    texts = ["" if text is None else str(text) for text in texts]

    features = model.vectorizer.transform(texts)
    dataset = SparseTfidfDataset(features, np.zeros((len(texts), len(model.labels))))
    loader = DataLoader(dataset, batch_size=int(model.config["batch_size"]), shuffle=False)

    model.network.eval()
    logits_all = []
    with torch.no_grad():
        for batch_features, _ in loader:
            batch_features = batch_features.to(model.device)
            logits_all.append(model.network(batch_features).cpu().numpy())

    logits = np.concatenate(logits_all)
    probabilities = 1 / (1 + np.exp(-logits))
    extra_threshold = float(model.config.get("extra_label_threshold", 0.8))
    predicted = decode_probabilities(probabilities, extra_threshold)

    return [
        [model.labels[index] for index, value in enumerate(row) if value]
        for row in predicted
    ]


def save_model(model: DeepLearningModel, output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path / "best_model.pt"

    torch.save(
        {
            "state_dict": model.network.state_dict(),
            "vectorizer": model.vectorizer,
            "config": model.config,
            "labels": model.labels,
            "history": model.history,
        },
        checkpoint_path,
    )

    with open(output_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                **{
                    k: v
                    for k, v in model.config.items()
                    if isinstance(v, (str, int, float, list, tuple, bool))
                },
                "ngram_range": list(model.config["ngram_range"]),
                "labels": model.labels,
                "train_time_seconds": model.history.get("train_time_seconds"),
                "best_val_f1_macro": model.history.get("best_val_f1_macro"),
                "train_loss": model.history.get("train_loss"),
                "validation_loss": model.history.get("validation_loss"),
                "validation_metrics": model.history.get("validation_metrics"),
            },
            f,
            indent=2,
        )

    print(f"Saved model to {checkpoint_path}")
    return checkpoint_path


def load_model(model_path: str | Path, device: str | None = None) -> DeepLearningModel:
    path = Path(model_path)
    if path.is_dir():
        path = path / "best_model.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"No saved model at {path}. Train first with: "
            ".venv/bin/python src/deep_learning.py"
        )

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    device = device or get_device_name()
    config = checkpoint["config"]

    network = TfidfMLP(
        input_dim=int(config["input_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        num_labels=len(checkpoint["labels"]),
        dropout=float(config["dropout"]),
    )
    network.load_state_dict(checkpoint["state_dict"])
    network.to(device)
    network.eval()

    return DeepLearningModel(
        network=network,
        vectorizer=checkpoint["vectorizer"],
        config={**config, "device": device},
        device=device,
        labels=list(checkpoint["labels"]),
        history=checkpoint.get("history", {}),
    )


def evaluate_test(
    model: DeepLearningModel,
    test_path: str,
    output_dir: str,
) -> dict:
    test_df = load_split(test_path)
    start = time.perf_counter()
    predicted_labels = predict(model, test_df["text"])
    inference_time = time.perf_counter() - start

    y_true = np.vstack(test_df["label_ids"].to_numpy()).astype(int)
    y_pred = np.array(
        [
            [int(label in predicted) for label in LABELS]
            for predicted in predicted_labels
        ]
    )
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "f1_micro": precision_recall_fscore_support(
            y_true, y_pred, average="micro", zero_division=0
        )[2],
        "inference_time_seconds": inference_time,
        "inference_time_per_example_ms": inference_time / max(len(test_df), 1) * 1000,
    }

    print("\nFinal Test Evaluation...")
    print(json.dumps({k: float(v) for k, v in metrics.items()}, indent=2))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "text": test_df["text"],
            "true_label": test_df["label"].map(str),
            "predicted_label": [str(labels) for labels in predicted_labels],
        }
    ).to_csv(output_path / "test_predictions.csv", index=False)

    with open(output_path / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

    print(f"Predictions saved to {output_path / 'test_predictions.csv'}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the TextBench TF-IDF + MLP classifier."
    )
    parser.add_argument("--train", default="data/processed/train.csv")
    parser.add_argument("--validation", default="data/processed/validation.csv")
    parser.add_argument("--test", default="data/processed/test.csv")
    parser.add_argument("--output", default="results/deep_learning")
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate a saved model on the test set. Does not train.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.evaluate_test:
        model = load_model(output_dir)
        print(f"Device: {model.device}")
        print(f"Loading model from: {output_dir / 'best_model.pt'}")
        print("Skipping training.")
        evaluate_test(model, args.test, args.output)
        return

    model = train(args.train, args.validation)
    save_model(model, output_dir)


if __name__ == "__main__":
    main()
