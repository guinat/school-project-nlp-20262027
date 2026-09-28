"""Common evaluation protocol for the three TextBench systems.

Every system is scored by exactly the same functions, on the same test set,
from its predictions written as lists of genres (``list[list[str]]``). The
functions below are pure: they never train, load or call a model. The only
exception is ``time_inference``, which receives a prediction function.

Conventions
-----------
- A prediction is a list of genres, possibly empty (a thresholded multi-label
  model may predict nothing).
- Accuracy is the exact match (subset accuracy): an example counts as correct
  only if the predicted set equals the true set.
- Precision, recall and F1 are macro averages over the 10 genres, computed on
  the binary indicator matrices, with zero_division=0.
"""

import ast
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    multilabel_confusion_matrix,
    precision_recall_fscore_support,
)

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
NO_LABEL = "(aucun)"
SEED = 42


# Stage 1 - Label handling
def parse_labels(value) -> list[str]:
    """Turn "['sports']", ['sports'], 'sports', '[]' or NaN into a list of genres.

    Unlike the parsers of the three systems, an empty list is accepted: it is a
    valid prediction (the transformer can predict nothing), never a valid truth.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, str):
        value = value.strip()
        parsed = ast.literal_eval(value) if value.startswith("[") else [value]
    else:
        parsed = value
    if isinstance(parsed, str):
        parsed = [parsed]

    labels = [str(label).strip().lower() for label in parsed]
    unknown = sorted(set(labels) - set(LABELS))
    if unknown:
        raise ValueError(f"Unknown labels: {unknown}. Expected: {LABELS}")
    return labels


def to_indicator(label_lists) -> np.ndarray:
    """list[list[str]] -> binary matrix (n_examples, 10), columns in LABELS order."""
    return np.array(
        [[int(label in set(labels)) for label in LABELS] for labels in label_lists],
        dtype=int,
    ).reshape(-1, len(LABELS))


def from_indicator(matrix) -> list[list[str]]:
    return [[LABELS[j] for j in np.flatnonzero(row)] for row in np.asarray(matrix)]


def _as_indicator(y) -> np.ndarray:
    if isinstance(y, np.ndarray) and y.ndim == 2:
        return y.astype(int)
    return to_indicator([parse_labels(labels) for labels in y])


def _as_lists(y) -> list[list[str]]:
    if isinstance(y, np.ndarray) and y.ndim == 2:
        return from_indicator(y)
    return [parse_labels(labels) for labels in y]


def load_split(path) -> pd.DataFrame:
    """Read a processed split: columns text and labels (list of genres)."""
    df = pd.read_csv(path)
    return pd.DataFrame(
        {
            "text": df["summary"].fillna("").astype(str),
            "labels": df["genre"].apply(parse_labels),
        }
    )


# Stage 2 - Metrics
def evaluate(y_true, y_pred) -> dict:
    """The metrics of the course, identical for the three systems."""
    true = _as_indicator(y_true)
    pred = _as_indicator(y_pred)
    if true.shape != pred.shape:
        raise ValueError(f"Shape mismatch: {true.shape} vs {pred.shape}")

    precision, recall, f1, _ = precision_recall_fscore_support(
        true, pred, average="macro", zero_division=0
    )
    n_predicted = pred.sum(axis=1)
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "f1_micro": float(
            precision_recall_fscore_support(
                true, pred, average="micro", zero_division=0
            )[2]
        ),
        "n_examples": int(len(true)),
        "n_empty_predictions": int((n_predicted == 0).sum()),
        "n_multi_predictions": int((n_predicted > 1).sum()),
        "labels_per_example": float(n_predicted.mean()),
    }


def per_class_report(y_true, y_pred) -> pd.DataFrame:
    """Precision, recall, F1 and support per genre, plus the one-vs-rest counts
    (TP, FP, FN, TN) of sklearn's multilabel_confusion_matrix."""
    true = _as_indicator(y_true)
    pred = _as_indicator(y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        true, pred, average=None, labels=range(len(LABELS)), zero_division=0
    )
    mcm = multilabel_confusion_matrix(true, pred)
    return pd.DataFrame(
        {
            "genre": LABELS,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "tp": mcm[:, 1, 1],
            "fp": mcm[:, 0, 1],
            "fn": mcm[:, 1, 0],
            "tn": mcm[:, 0, 0],
        }
    )


def top1(y_pred, scores=None) -> list:
    """The single genre each system bets on, or None if it predicts nothing.

    With scores (n_examples, 10), it is the predicted genre with the highest
    score; ties go to the genre listed first. Without scores, it is the first
    genre of the list, which is only right if the system sorts its output by
    score (the baseline does, the deep learning model lists genres
    alphabetically).
    """
    result = []
    for i, labels in enumerate(_as_lists(y_pred)):
        if not labels:
            result.append(None)
        elif scores is None:
            result.append(labels[0])
        else:
            result.append(
                max(labels, key=lambda g: (scores[i][LABEL2ID[g]], -labels.index(g)))
            )
    return result


def confusion_matrix(y_true, y_pred, scores=None) -> pd.DataFrame:
    """Confusion matrix 10 x 11 on the single-label examples.

    Rows: true genre. Columns: the 10 genres plus NO_LABEL for empty
    predictions. Multi-label predictions are projected to one column:
    - empty prediction            -> NO_LABEL;
    - true genre among predicted  -> diagonal (the example is found);
    - otherwise                   -> top-1 of the system (see ``top1``).
    Multi-label truths (6 examples in test) are left out, because a row must
    count one example once.
    """
    true_lists = _as_lists(y_true)
    pred_lists = _as_lists(y_pred)
    best = top1(pred_lists, scores)
    columns = LABELS + [NO_LABEL]
    matrix = pd.DataFrame(0, index=LABELS, columns=columns)
    for truth, predicted, bet in zip(true_lists, pred_lists, best):
        if len(truth) != 1:
            continue
        genre = truth[0]
        if not predicted:
            matrix.loc[genre, NO_LABEL] += 1
        elif genre in predicted:
            matrix.loc[genre, genre] += 1
        else:
            matrix.loc[genre, bet] += 1
    matrix.index.name = "vrai"
    matrix.columns.name = "prédit"
    return matrix


# Stage 3 - Decoding (scores -> genres)
def decode_top_threshold(probabilities, threshold: float) -> list[list[str]]:
    """Common decoding: the most probable genre, plus every other genre whose
    probability reaches ``threshold``. Never empty. Genres are sorted by
    decreasing probability, so the first one is the top-1."""
    decoded = []
    for row in np.asarray(probabilities, dtype=float):
        order = np.argsort(-row, kind="stable")
        keep = [order[0]] + [j for j in order[1:] if row[j] >= threshold]
        decoded.append([LABELS[j] for j in keep])
    return decoded


def decode_independent(probabilities, threshold: float = 0.5) -> list[list[str]]:
    """Independent threshold per genre (the transformer as delivered). Can be empty."""
    return from_indicator(np.asarray(probabilities) >= threshold)


def select_threshold(y_true, decode_fn, grid, metric: str = "f1_macro") -> tuple:
    """Pick the value of ``grid`` that maximises ``metric`` on the data given.

    Must only be called on validation data. Ties go to the first value of the
    grid, so the order of the grid is part of the protocol.
    """
    rows = []
    for value in grid:
        metrics = evaluate(y_true, decode_fn(value))
        rows.append({"threshold": value, **metrics})
    table = pd.DataFrame(rows)
    best = table.loc[table[metric].idxmax(), "threshold"]
    return best, table


# Stage 4 - Robustness
def bootstrap_f1_macro(y_true, y_pred, n_boot: int = 1000, seed: int = SEED,
                       alpha: float = 0.05) -> dict:
    """Percentile bootstrap CI of the F1 macro, resampling test examples."""
    true = _as_indicator(y_true)
    pred = _as_indicator(y_pred)
    rng = np.random.default_rng(seed)
    n = len(true)
    values = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        values[b] = precision_recall_fscore_support(
            true[idx], pred[idx], average="macro", zero_division=0
        )[2]
    low, high = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return {
        "f1_macro": float(
            precision_recall_fscore_support(
                true, pred, average="macro", zero_division=0
            )[2]
        ),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_boot": n_boot,
        "seed": seed,
    }


def paired_bootstrap_difference(y_true, pred_a, pred_b, n_boot: int = 1000,
                                seed: int = SEED, alpha: float = 0.05) -> dict:
    """CI of F1_macro(a) - F1_macro(b), both systems scored on the same resamples."""
    true = _as_indicator(y_true)
    a = _as_indicator(pred_a)
    b = _as_indicator(pred_b)
    rng = np.random.default_rng(seed)
    n = len(true)

    def f1(t, p):
        return precision_recall_fscore_support(t, p, average="macro", zero_division=0)[2]

    diffs = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[k] = f1(true[idx], a[idx]) - f1(true[idx], b[idx])
    low, high = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return {
        "difference": float(f1(true, a) - f1(true, b)),
        "ci_low": float(low),
        "ci_high": float(high),
        "share_a_better": float((diffs > 0).mean()),
        "n_boot": n_boot,
        "seed": seed,
    }


def mcnemar_test(y_true, pred_a, pred_b) -> dict:
    """Exact McNemar test on the exact-match correctness of two systems.

    b = examples only system a gets right, c = only system b gets right.
    Under H0 (same error rate), b ~ Binomial(b + c, 0.5).
    """
    from scipy.stats import binomtest

    true = _as_indicator(y_true)
    right_a = (_as_indicator(pred_a) == true).all(axis=1)
    right_b = (_as_indicator(pred_b) == true).all(axis=1)
    b = int((right_a & ~right_b).sum())
    c = int((~right_a & right_b).sum())
    p_value = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return {
        "both_right": int((right_a & right_b).sum()),
        "only_a_right": b,
        "only_b_right": c,
        "both_wrong": int((~right_a & ~right_b).sum()),
        "p_value_exact": float(p_value),
    }


# Stage 5 - Timing and final table
def time_inference(predict_fn, texts, n_repeats: int = 3, warmup: int = 32,
                   device: str = "cpu", batch_size=None) -> dict:
    """Wall-clock time of ``predict_fn(texts)`` under the same conditions for
    every system: same texts, one warm-up call on the first ``warmup`` texts
    (loads kernels and caches, not counted), then ``n_repeats`` timed passes
    over all texts. The median pass is reported.

    ``predict_fn`` must return host (CPU) objects, so that an asynchronous
    device like MPS has finished when the clock stops.
    """
    texts = list(texts)
    if warmup:
        predict_fn(texts[:warmup])
    durations = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        predict_fn(texts)
        durations.append(time.perf_counter() - start)
    median = float(np.median(durations))
    return {
        "device": device,
        "batch_size": batch_size,
        "warmup_examples": warmup,
        "n_repeats": n_repeats,
        "n_examples": len(texts),
        "total_seconds_median": median,
        "total_seconds_min": float(min(durations)),
        "total_seconds_max": float(max(durations)),
        "ms_per_example": median / max(len(texts), 1) * 1000,
    }


TABLE_COLUMNS = ["Méthode", "Accuracy", "Précision", "Rappel", "F1 macro",
                 "Train", "Inférence"]


def comparison_table(rows) -> pd.DataFrame:
    """Final table of the template (section 8.4).

    ``rows``: iterable of dicts with keys name, metrics (output of
    ``evaluate``), train_seconds and inference_ms (ms per example).
    """
    records = []
    for row in rows:
        metrics = row["metrics"]
        records.append(
            {
                "Méthode": row["name"],
                "Accuracy": metrics["accuracy"],
                "Précision": metrics["precision_macro"],
                "Rappel": metrics["recall_macro"],
                "F1 macro": metrics["f1_macro"],
                "Train": row.get("train_seconds"),
                "Inférence": row.get("inference_ms"),
            }
        )
    return pd.DataFrame(records, columns=TABLE_COLUMNS)


def format_table(table: pd.DataFrame) -> pd.DataFrame:
    """Human-readable copy: 3 decimals, train in s, inference in ms/example."""
    out = table.copy()
    for col in ["Accuracy", "Précision", "Rappel", "F1 macro"]:
        out[col] = out[col].map(lambda v: f"{v:.3f}")
    out["Train"] = out["Train"].map(
        lambda v: "–" if v is None or pd.isna(v) else ("0 s" if v == 0 else f"{v:.1f} s")
    )
    out["Inférence"] = out["Inférence"].map(
        lambda v: "–" if v is None or pd.isna(v) else f"{v:.2f} ms/ex."
    )
    return out


def to_markdown(df: pd.DataFrame, index: bool = False, floatfmt: str = ".3f") -> str:
    """Minimal Markdown table writer (pandas.to_markdown needs tabulate)."""
    frame = df.reset_index() if index else df

    def cell(value):
        if isinstance(value, (float, np.floating)):
            return format(value, floatfmt)
        return str(value)

    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    separator = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, separator, *body]) + "\n"


# Stage 6 - Figures
# Categorical colours follow the system, never its rank (validated reference
# palette, fixed order). Sequential blue ramp for the confusion heatmaps.
SYSTEM_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def plot_confusion_matrices(matrices: dict, path=None, ncols=None):
    """One heatmap per system. Colour = share of the row (true genre), text =
    count, so small classes stay readable next to large ones."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    blues = LinearSegmentedColormap.from_list(
        "blues", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    ncols = ncols or len(matrices)
    nrows = int(np.ceil(len(matrices) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.4 * nrows),
                             squeeze=False)
    for ax in axes.ravel()[len(matrices):]:
        ax.axis("off")
    for ax, (name, matrix) in zip(axes.ravel(), matrices.items()):
        counts = matrix.to_numpy()
        share = counts / np.clip(counts.sum(axis=1, keepdims=True), 1, None)
        ax.imshow(share, cmap=blues, vmin=0, vmax=1)
        for i in range(counts.shape[0]):
            for j in range(counts.shape[1]):
                if counts[i, j]:
                    ax.text(j, i, counts[i, j], ha="center", va="center", fontsize=7,
                            color="white" if share[i, j] > 0.5 else INK)
        ax.set_xticks(range(counts.shape[1]), matrix.columns, rotation=60,
                      ha="right", fontsize=7.5)
        ax.set_yticks(range(counts.shape[0]),
                      [f"{g} ({n})" for g, n in zip(matrix.index, counts.sum(axis=1))],
                      fontsize=7.5)
        ax.set_xlabel("prédit (top-1 si erreur)", fontsize=8, color=INK_MUTED)
        ax.set_ylabel("vrai genre (effectif)", fontsize=8, color=INK_MUTED)
        ax.set_title(name, fontsize=9.5, color=INK)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    return fig


def plot_per_class_f1(reports: dict, path=None):
    """Grouped bars: F1 of each genre for each system, support under the label."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.5, 3.3))
    n = len(reports)
    slot = 0.84 / n
    x = np.arange(len(LABELS))
    support = next(iter(reports.values()))["support"].to_numpy()
    for k, (name, report) in enumerate(reports.items()):
        # 2 px of white between neighbouring bars.
        positions = x + (k - (n - 1) / 2) * slot
        ax.bar(positions, report["f1"], slot, label=name,
               color=SYSTEM_COLORS[k], edgecolor="white", linewidth=1.0)
        # A zero F1 draws no bar: say it, so it does not read as missing data.
        for pos, value in zip(positions, report["f1"]):
            if value == 0:
                ax.text(pos, 0.01, "0", ha="center", va="bottom", fontsize=7,
                        color=INK_MUTED)
    ax.set_xticks(x, [f"{g}\nn={s}" for g, s in zip(LABELS, support)], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 (test)", fontsize=8.5, color=INK_MUTED)
    _style(ax)
    ax.legend(ncol=n, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.16),
              frameon=False)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    return fig


def plot_loss_curves(curves: dict, path=None):
    """curves: {title: {"epochs", "train", "validation", "best_epoch"}}.
    One panel per model: the two losses are not on the same scale."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, axes = plt.subplots(1, len(curves), figsize=(4.8 * len(curves), 3.0),
                             squeeze=False)
    for ax, (title, curve) in zip(axes.ravel(), curves.items()):
        ax.plot(curve["epochs"], curve["train"], "o-", lw=2, ms=4.5,
                color=SYSTEM_COLORS[0], label="train")
        ax.plot(curve["epochs"], curve["validation"], "s-", lw=2, ms=4.5,
                color=SYSTEM_COLORS[1], label="validation")
        if curve.get("best_epoch"):
            ax.axvline(curve["best_epoch"], ls="--", color=INK_MUTED, lw=1,
                       label="modèle retenu")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("époque", fontsize=8.5, color=INK_MUTED)
        ax.set_ylabel("loss", fontsize=8.5, color=INK_MUTED)
        ax.set_title(title, fontsize=9, color=INK)
        _style(ax)
        ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    return fig


def save_table(table: pd.DataFrame, stem, index: bool = False) -> None:
    """Write the same table as CSV and Markdown."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(stem.with_suffix(".csv"), index=index)
    stem.with_suffix(".md").write_text(to_markdown(table, index=index), encoding="utf-8")
