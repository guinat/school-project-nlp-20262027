"""Run the three systems once and store everything the common evaluation needs.

Stage "systems" (slow, ~2 min, needs the saved models):
  - baseline: rerun on validation and test, keyword scores saved;
  - deep learning: probabilities of the delivered model (best_model.pt),
    plus retraining with seed 42 to check reproducibility (the delivered model
    is never overwritten);
  - transformer: probabilities of the delivered model (best_model), no
    retraining;
  - inference timing under the same conditions for the three systems.
Stage "analysis" (fast, only reads results/evaluation/): tables, figures,
bootstrap, McNemar, error selection.
Stage "history" (optional, ~10 s, needs git): retrains the first version of the
deep learning model, which used an independent 0.5 threshold, to document why
its decoding rule was changed.

Usage:
  .venv/bin/python src/run_evaluation.py              # systems + analysis
  .venv/bin/python src/run_evaluation.py --stage analysis
  .venv/bin/python src/run_evaluation.py --stage history
"""

import argparse
import json
import os
import platform
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluation as ev  # noqa: E402
from evaluation import LABELS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
OUT = RESULTS / "evaluation"
SCORES = OUT / "scores"

SPLITS = ["validation", "test"]
SYSTEMS = {
    "baseline": "Baseline (règles)",
    "deep_learning": "Deep Learning (TF-IDF + MLP)",
    "transformer": "Transformer (DistilBERT)",
}
INFERENCE_BATCH_SIZE = 32
# Goodreads "show less" button left in the summaries by the scraping.
LESS_ARTIFACT = re.compile(r"\s*\(less\)")


def split_data(split: str) -> pd.DataFrame:
    return ev.load_split(DATA / f"{split}.csv")


def write_json(data, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=float)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_scores(system: str, split: str) -> np.ndarray:
    return pd.read_csv(SCORES / f"{system}_{split}.csv")[LABELS].to_numpy()


def save_scores(scores: np.ndarray, system: str, split: str) -> None:
    SCORES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(scores, columns=LABELS).to_csv(SCORES / f"{system}_{split}.csv",
                                                index=False)


# Adapters: each one turns a system into scores (n, 10) or genres
def baseline_scores(texts) -> np.ndarray:
    import baseline as bl

    return np.array([[bl.score_text(t)[g] for g in LABELS] for t in texts])


def baseline_decode(scores: np.ndarray, ratio: float) -> list[list[str]]:
    """The baseline's own decoding (tie-break, default genre) with another
    ratio. np.inf keeps the top genre only."""
    import baseline as bl

    previous = bl.EXTRA_LABEL_RATIO
    bl.EXTRA_LABEL_RATIO = ratio
    try:
        return [bl.decode_scores(dict(zip(LABELS, row))) for row in scores]
    finally:
        bl.EXTRA_LABEL_RATIO = previous


def dl_probabilities(model, texts, batch_size: int = INFERENCE_BATCH_SIZE) -> np.ndarray:
    import torch

    features = model.vectorizer.transform(list(texts))
    model.network.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            batch = torch.from_numpy(features[start:start + batch_size].toarray()).float()
            chunks.append(torch.sigmoid(model.network(batch.to(model.device))).cpu().numpy())
    return np.concatenate(chunks)


def dl_decode(probabilities: np.ndarray, threshold: float = 0.8) -> list[list[str]]:
    """Delivered DL decoding (argmax + p >= 0.8), genres sorted by probability."""
    return ev.decode_top_threshold(probabilities, threshold)


def load_transformer(device: str = "cpu"):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = RESULTS / "transformer" / "best_model"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device).eval()
    return model, tokenizer


def transformer_probabilities(model, tokenizer, texts, device: str = "cpu",
                              batch_size: int = INFERENCE_BATCH_SIZE,
                              max_length: int = 256) -> np.ndarray:
    """Sigmoid of the logits, fp32. Same tokenizer settings as training."""
    import torch

    texts = list(texts)
    chunks = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(texts[start:start + batch_size], truncation=True,
                                max_length=max_length, padding=True,
                                return_tensors="pt").to(device)
            logits = model(**encoded).logits.float()
            chunks.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(chunks)


def same_sets(a, b) -> float:
    return float(np.mean([set(x) == set(y) for x, y in zip(a, b)]))


# Stage "systems"
def run_baseline(log: dict) -> None:
    import baseline as bl

    for split in SPLITS:
        df = split_data(split)
        scores = baseline_scores(df["text"])
        save_scores(scores, "baseline", split)
        rerun = bl.predict(df["text"])
        delivered_path = RESULTS / "baseline" / f"{split}_predictions.csv"
        delivered = pd.read_csv(delivered_path)["predicted_label"].map(ev.parse_labels)
        log[f"baseline_{split}_rerun_identical"] = same_sets(rerun, delivered)
        # Order matters for the baseline (first genre = top-1): check it too.
        log[f"baseline_{split}_rerun_same_order"] = float(
            np.mean([list(a) == list(b) for a, b in zip(rerun, delivered)]))
        log[f"baseline_{split}_decode_from_scores_identical"] = same_sets(
            baseline_decode(scores, bl.EXTRA_LABEL_RATIO), delivered)
        log[f"baseline_{split}_no_rule_fired"] = int((scores.max(axis=1) == 0).sum())


def run_deep_learning(log: dict, retrain_runs: int) -> None:
    import deep_learning as dl
    import torch

    model = dl.load_model(RESULTS / "deep_learning", device="cpu")
    for split in SPLITS:
        probabilities = dl_probabilities(model, split_data(split)["text"])
        save_scores(probabilities, "deep_learning", split)
    delivered = pd.read_csv(RESULTS / "deep_learning" / "test_predictions.csv")
    log["deep_learning_test_reload_identical"] = same_sets(
        dl_decode(load_scores("deep_learning", "test")),
        delivered["predicted_label"].map(ev.parse_labels))

    # Reproducibility: retrain with seed 42 in memory, never overwrite
    # results/deep_learning/best_model.pt.
    test = split_data("test")
    validation = split_data("validation")
    reference = load_scores("deep_learning", "test")
    runs = []
    for run in range(retrain_runs):
        retrained = dl.train(DATA / "train.csv", DATA / "validation.csv")
        test_probabilities = dl_probabilities(retrained, test["text"])
        predictions = dl_decode(test_probabilities)
        runs.append({
            "run": run + 1,
            "device": retrained.device,
            "train_seconds": retrained.history["train_time_seconds"],
            "epochs_run": len(retrained.history["train_loss"]),
            "best_epoch": int(np.argmax([m["f1_macro"] for m in
                                         retrained.history["validation_metrics"]]) + 1),
            "validation": ev.evaluate(validation["labels"],
                                      dl_decode(dl_probabilities(retrained, validation["text"]))),
            "test": ev.evaluate(test["labels"], predictions),
            "test_agreement_with_delivered": same_sets(predictions, dl_decode(reference)),
            "test_max_abs_probability_gap": float(np.abs(test_probabilities - reference).max()),
        })
        del retrained
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    write_json(runs, OUT / "dl_reproducibility.json")


def run_transformer(log: dict) -> None:
    model, tokenizer = load_transformer("cpu")
    for split in SPLITS:
        probabilities = transformer_probabilities(model, tokenizer, split_data(split)["text"])
        save_scores(probabilities, "transformer", split)
    delivered = pd.read_csv(RESULTS / "transformer" / "test_predictions.csv")
    log["transformer_test_fp32_cpu_identical_to_delivered"] = same_sets(
        ev.decode_independent(load_scores("transformer", "test")),
        delivered["predicted_label"].map(ev.parse_labels))
    config = read_json(RESULTS / "transformer" / "config.json")
    log["transformer_validation_accuracy_delivered_config"] = config["validation_metrics"]["eval_accuracy"]
    log["transformer_validation_accuracy_recomputed"] = ev.evaluate(
        split_data("validation")["labels"],
        ev.decode_independent(load_scores("transformer", "validation")))["accuracy"]
    # Share of summaries cut by the 256-token limit.
    lengths = [len(tokenizer(t, truncation=False)["input_ids"])
               for t in split_data("test")["text"]]
    log["transformer_test_truncated_share"] = float(np.mean(np.array(lengths) > 256))
    log["transformer_test_median_tokens"] = float(np.median(lengths))


def run_artifact_check() -> None:
    """Transformer probabilities on the test summaries without "(less)".

    The artifact's frequency depends on the genre (source of the scraping), so
    a model could use it as a shortcut. Only the transformer can see it: "less"
    is not in the baseline lexicon and is an English stop word for the DL
    vectoriser, so their predictions cannot change.
    """
    model, tokenizer = load_transformer("cpu")
    cleaned = split_data("test")["text"].str.replace(LESS_ARTIFACT, "", regex=True)
    save_scores(transformer_probabilities(model, tokenizer, cleaned),
                "transformer", "test_sans_less")


def run_timing(log: dict) -> None:
    """Same texts (test), same warm-up, same repeats, same batch size for the
    two networks. CPU for the three systems (common condition), MPS in
    addition for the two networks."""
    import baseline as bl
    import deep_learning as dl
    import torch

    texts = split_data("test")["text"].tolist()
    timings = []

    timings.append({"system": "baseline", **ev.time_inference(
        bl.predict, texts, device="cpu", batch_size=1)})

    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    for device in devices:
        dl_model = dl.load_model(RESULTS / "deep_learning", device=device)
        timings.append({"system": "deep_learning", **ev.time_inference(
            lambda xs: dl_decode(dl_probabilities(dl_model, xs)), texts,
            device=device, batch_size=INFERENCE_BATCH_SIZE)})

        tr_model, tokenizer = load_transformer(device)
        timings.append({"system": "transformer", **ev.time_inference(
            lambda xs: ev.decode_independent(
                transformer_probabilities(tr_model, tokenizer, xs, device)),
            texts, device=device, batch_size=INFERENCE_BATCH_SIZE)})

    write_json({
        "machine": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "protocol": ("508 test summaries, 1 warm-up call on 32 texts (not timed), "
                     "3 timed passes, median reported; end-to-end: raw text -> genres"),
        "timings": timings,
    }, OUT / "timing.json")


def run_costs() -> None:
    import baseline as bl
    import deep_learning as dl

    model = dl.load_model(RESULTS / "deep_learning", device="cpu")
    transformer_dir = RESULTS / "transformer" / "best_model"
    from transformers import AutoModelForSequenceClassification

    transformer = AutoModelForSequenceClassification.from_pretrained(transformer_dir)
    write_json({
        "baseline": {
            "learned_parameters": 0,
            "keywords": sum(len(g["strong"]) + len(g["weak"]) for g in bl.RULES.values()),
            "disk_bytes": (ROOT / "src" / "baseline.py").stat().st_size,
            "disk_note": "src/baseline.py (lexique compris)",
        },
        "deep_learning": {
            "learned_parameters": int(sum(p.numel() for p in model.network.parameters())),
            "vocabulary": int(len(model.vectorizer.vocabulary_)),
            "disk_bytes": (RESULTS / "deep_learning" / "best_model.pt").stat().st_size,
            "disk_note": "best_model.pt (poids + vectoriseur TF-IDF)",
        },
        "transformer": {
            "learned_parameters": int(sum(p.numel() for p in transformer.parameters())),
            "disk_bytes": (transformer_dir / "model.safetensors").stat().st_size,
            "disk_note": "best_model/model.safetensors (fp32)",
        },
    }, OUT / "costs.json")


def run_systems(retrain_runs: int = 2, timing: bool = True) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    log = {}
    run_baseline(log)
    run_transformer(log)
    run_deep_learning(log, retrain_runs)
    run_artifact_check()
    run_costs()
    if timing:
        run_timing(log)
    write_json(log, OUT / "checks.json")
    print(json.dumps(log, indent=2))


# First version of src/deep_learning.py: independent 0.5 threshold per genre,
# used both for early stopping and for the answers. Replaced in 8e5e5a8.
FIRST_DL_COMMIT = "5cd4b7b"


def run_dl_history() -> None:
    """Retrain the first version of the DL model to document why its decoding
    rule was changed. The code is read from git history and trained in memory:
    nothing is written in results/deep_learning/."""
    import importlib.util
    import subprocess
    import tempfile

    source = subprocess.run(
        ["git", "show", f"{FIRST_DL_COMMIT}:src/deep_learning.py"],
        cwd=ROOT, capture_output=True, text=True, check=True).stdout
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "deep_learning_v1.py"
        path.write_text(source, encoding="utf-8")
        spec = importlib.util.spec_from_file_location("deep_learning_v1", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.train(DATA / "train.csv", DATA / "validation.csv")

    validation = split_data("validation")
    probabilities = dl_probabilities(model, validation["text"])
    first = ev.evaluate(validation["labels"], ev.decode_independent(probabilities, 0.5))
    n = first["n_examples"]
    write_json({
        "commit": FIRST_DL_COMMIT,
        "rule": "un genre des que sa probabilite atteint 0,5 (regle independante)",
        "device": model.device,
        "best_epoch": int(np.argmax([m["f1_macro"] for m in
                                     model.history["validation_metrics"]]) + 1),
        "epochs_run": len(model.history["train_loss"]),
        "validation": first,
        "validation_empty_share": first["n_empty_predictions"] / n,
        "validation_multi_share": first["n_multi_predictions"] / n,
        "validation_true_multi_share": float(np.mean([len(t) > 1 for t in validation["labels"]])),
    }, OUT / "dl_premiere_version.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", choices=["systems", "analysis", "history", "all"],
                        default="all")
    parser.add_argument("--retrain-runs", type=int, default=2)
    parser.add_argument("--skip-timing", action="store_true")
    args = parser.parse_args()
    start = time.perf_counter()
    if args.stage in ("systems", "all"):
        run_systems(args.retrain_runs, timing=not args.skip_timing)
    if args.stage == "history":
        run_dl_history()
    if args.stage in ("analysis", "all"):
        from analysis import run_analysis

        run_analysis()
    print(f"Done in {time.perf_counter() - start:.0f} s")
