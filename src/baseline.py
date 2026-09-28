import argparse
import ast
import html
import json
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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

# Weight of a keyword that is (almost) specific to one genre vs. a keyword that
# only makes a genre more likely. Each distinct keyword counts once per summary:
# repeating "vampire" twenty times should not beat twenty different clues.
STRONG_WEIGHT = 2.5
WEAK_WEIGHT = 1.0

# Fallback when no rule fires at all: the majority genre of the training set.
DEFAULT_LABEL = "thriller"

# A second (or third) genre is kept only if it stays close to the best score.
EXTRA_LABEL_RATIO = 0.9

# Scope of a negation cue, in words. Only used when negation handling is on.
NEGATION_WINDOW = 4

# Ties are broken with this order: genres with a narrow, specific vocabulary
# first, generic ones last. Without it, "thriller" would win every tie.
TIE_BREAK = [
    "sports",
    "travel",
    "psychology",
    "romance",
    "science",
    "horror",
    "fantasy",
    "crime",
    "history",
    "thriller",
]


# Stage 1 - Preprocessing
def normalize_text(text) -> str:
    """Undo HTML entities and collapse whitespace, then lowercase.

    The course insists that every cleaning step must be justified, because
    removing noise can also remove signal. Here nothing is deleted except
    markup and URLs, which carry no genre information.
    """
    text = html.unescape("" if text is None else str(text))
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"<[^>]{1,40}>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


NEGATION_CUES = re.compile(
    r"\b(?:not|n't|never|no|none|nothing|without|hardly|barely|scarcely|"
    r"neither|nor|cannot|isn|aren|wasn|weren|doesn|don|didn|won)\b"
)


def strip_negated_scopes(text: str, window: int = NEGATION_WINDOW) -> str:
    """Blank out the words that follow a negation cue.

    The course shows that deleting "ne" and "pas" destroys the very signal a
    sentiment classifier looks for. The mirror problem for a lexicon is that
    "this is not a love story" scores romance. Scope ends at the first strong
    punctuation or after `window` words, whichever comes first.
    """
    pieces, position = [], 0
    for cue in NEGATION_CUES.finditer(text):
        pieces.append(text[position:cue.end()])
        tail = text[cue.end():]
        stop = len(tail)
        words = list(re.finditer(r"\S+", tail))
        if len(words) > window:
            stop = min(stop, words[window].start())
        punctuation = re.search(r"[.;:!?]", tail)
        if punctuation:
            stop = min(stop, punctuation.start())
        pieces.append(" " * stop)
        position = cue.end() + stop
    pieces.append(text[position:])
    return "".join(pieces)


def preprocess(text, negation: bool = False) -> str:
    text = normalize_text(text)
    if negation:
        text = strip_negated_scopes(text)
    return text


# Stage 2 - Representation
# Keyword lexicon. Terms are small regular expressions matched on word
# boundaries, so "murder" also covers "murders" / "murdered" / "murderer".
# This suffix trick is the heuristic stemming of the course, not lemmatisation.
# Multi-word terms ("world war", "serial killer") are the bigrams of the course.
RULES = {
    "crime": {
        "strong": [
            "detective", "inspector", "homicide", "murder(?:s|ed|er|ers)?",
            "alibi", "blackmail", "robbery", "burglary", "forensic",
            "precinct", r"interrogat\w+", "culprit", "scotland yard",
            "private eye",
        ],
        "weak": [
            "police", r"investigat\w+", "evidence", "clues?", "witness(?:es)?",
            "victims?", "crimes?", "criminal", r"confess\w+", "motive",
            "guilty", "arrest(?:ed|s)?", "suspects?", "stabbed", "theft",
            "stolen", "trial", "jury",
        ],
    },
    "fantasy": {
        "strong": [
            "wizard", r"sorcer\w+", "dragons?", "magical?", "spells?",
            r"enchant\w+", "prophecy", "elves", "elf", "dwarves", "orcs?",
            "warlock", "mage", "runes?", "witchcraft", "fantasy",
        ],
        "weak": [
            "kingdoms?", "throne", "swords?", "witch(?:es)?", "kings?",
            "queens?", "prince(?:ss)?", "realm", "quest", "castle", "crown",
            "gods?", "demons?", "curse[ds]?", "legends?", r"myth\w+",
            "warriors?", "destiny", "palace", "knights?",
        ],
    },
    "history": {
        "strong": [
            "napoleon", "world war", "civil war", "medieval", "roman empire",
            "dynasty", "colonial", "slavery", "regiment", "middle ages",
            "historical novel", "french revolution",
        ],
        "weak": [
            "century", "historical", "empire", "wars?", "soldiers?",
            "battles?", "troops", "frigate", "lieutenant", "colonel",
            "england", "france", "britain", "spain", "rome", "roman",
            r"aristocra\w+", "nobleman", r"monarch\w*",
        ],
    },
    "horror": {
        "strong": [
            "vampires?", "werewol(?:f|ves)", "zombies?", "haunt(?:ed|ing|s)?",
            "ghosts?", "undead", r"exorcis\w+", "demonic", "occult",
            "supernatural", "monsters?", "coffin", "crypt", "gruesome",
            "possessed",
        ],
        "weak": [
            "blood", "terror", "evil", "darkness", "corpses?", "graves?",
            "creatures?", "horror", "spirits?", r"scream\w*", "flesh",
            "chilling", "eerie", "nightmares?",
        ],
    },
    "psychology": {
        "strong": [
            r"psycholog\w+", r"psychiatr\w+", "cognitive", r"neuroscien\w+",
            "therapy", "mindfulness", "unconscious", "behavioral science",
        ],
        "weak": [
            "brains?", "mind", "emotions?", "behaviou?rs?", "habits?",
            "motivation", "thinking", "insights?", "stress", "anxiety",
            "patients?",
        ],
    },
    "romance": {
        "strong": [
            "romance", "romantic", "love story", "falls? in love",
            r"seduc\w+", "passionate", "heartbreak", "happily ever after",
        ],
        "weak": [
            "love", "heart", "boyfriend", "attraction", "desire", "couple",
            "dating", "chemistry", "passion", "bride", "wedding",
        ],
    },
    "science": {
        "strong": [
            "spaceships?", "starships?", r"galax\w+", "planets?", "aliens?",
            "interstellar", "astronaut", "robots?", "cyborg",
            "artificial intelligence", "quantum", "spacecraft",
            "extraterrestrial", "hyperspace", r"terraform\w+", "science fiction",
        ],
        "weak": [
            "space", "science", "scientists?", "technology", "universe",
            "earth", "experiments?", r"laborator\w+", "physics", "evolution",
            "species", "mars", "orbit", "humanity", "colony", "fleet",
            "nuclear",
        ],
    },
    "sports": {
        "strong": [
            "football", "basketball", "baseball", "hockey", "quarterback",
            "soccer", "athletes?", "olympics?", "tournament", "championship",
            "nfl", "nba", "tennis", r"wrestl\w+",
        ],
        "weak": [
            "teams?", "games?", "players?", "league", "season", "stadium",
            "coach", "fans", "sports?", "rookie", "score",
        ],
    },
    "thriller": {
        "strong": [
            "thriller", "suspense", "cia", "fbi", "assassins?", "terrorists?",
            "conspiracy", "espionage", "spy", "hostages?", r"kidnap\w+",
            "serial killer", "manhunt", "covert", "sniper", "mi5", "mi6",
        ],
        "weak": [
            "agents?", "mission", "government", "secrets?", r"danger\w*",
            "chase", "deadly", "operative", "killer",
        ],
    },
    "travel": {
        "strong": [
            r"travel\w*", r"backpack\w*", "expedition", "voyage", "pilgrimage",
            "guidebook", "wanderlust", "itinerary", "travell?er", "road trip",
        ],
        "weak": [
            "journey", "adventure", "continent", "landscape", "islands?",
            "mountains?", "desert", "countryside", "memoir", r"trek\w*",
            "tour", "miles",
        ],
    },
}


def compile_rules(rules: dict) -> dict:
    """Pre-compile one regex per genre and per strength (strong / weak)."""
    compiled = {}
    for genre, groups in rules.items():
        compiled[genre] = {}
        for strength, terms in groups.items():
            # Longest first so that "world war" wins over "wars?".
            ordered = sorted(terms, key=len, reverse=True)
            pattern = r"\b(?:" + "|".join(ordered) + r")\b"
            compiled[genre][strength] = re.compile(pattern, re.IGNORECASE)
    return compiled


COMPILED_RULES = compile_rules(RULES)


def score_text(text, negation: bool = False) -> dict[str, float]:
    """Weighted count of the distinct keywords matched by each genre.

    Counting distinct terms rather than occurrences turns the score into a
    measure of how many different clues the summary gives, and removes the
    length bias that a raw term-frequency count would introduce.
    """
    text = preprocess(text, negation)
    scores = {}
    for genre, patterns in COMPILED_RULES.items():
        strong = {match.lower() for match in patterns["strong"].findall(text)}
        weak = {match.lower() for match in patterns["weak"].findall(text)}
        scores[genre] = STRONG_WEIGHT * len(strong) + WEAK_WEIGHT * len(weak)
    return scores


# Stage 3 - Decision
def decode_scores(scores: dict[str, float]) -> list[str]:
    """Always return at least one genre; extra genres only if close to the best.

    The ratio is the decision threshold of the course: lowering it adds genres,
    so recall goes up and precision goes down.
    """
    best = max(scores.values())
    if best == 0:
        return [DEFAULT_LABEL]

    # Tie-break: among the genres reaching the best score, keep the most specific.
    top = min(
        (genre for genre, value in scores.items() if value == best),
        key=TIE_BREAK.index,
    )

    predicted = [top]
    for genre in LABELS:
        if genre == top:
            continue
        if scores[genre] >= EXTRA_LABEL_RATIO * best:
            predicted.append(genre)
    return predicted


def predict(texts, negation: bool = False) -> list[list[str]]:
    if isinstance(texts, pd.Series):
        texts = texts.tolist()
    if isinstance(texts, str):
        texts = [texts]

    return [decode_scores(score_text(text, negation)) for text in texts]


def explain(text: str, top_k: int = 3, negation: bool = False) -> dict:
    """Show which keywords fired, to inspect a prediction by hand."""
    cleaned = preprocess(text, negation)
    scores = score_text(text, negation)
    ranking = sorted(scores.items(), key=lambda item: -item[1])[:top_k]
    return {
        "prediction": decode_scores(scores),
        "top_scores": ranking,
        "matched_keywords": {
            genre: {
                strength: sorted({
                    match.lower()
                    for match in COMPILED_RULES[genre][strength].findall(cleaned)
                })
                for strength in ("strong", "weak")
            }
            for genre, _ in ranking
        },
    }


# Stage 4 - Evaluation
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


def to_indicator(label_lists) -> np.ndarray:
    return np.array(
        [[int(label in labels) for label in LABELS] for labels in label_lists]
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "f1_micro": precision_recall_fscore_support(
            y_true, y_pred, average="micro", zero_division=0
        )[2],
    }


def majority_reference(data) -> dict:
    """Always predict the majority genre.

    The course uses this reference to show that accuracy alone can hide a class
    the system never finds. It is the floor the rules have to beat.
    """
    df = load_split(data)
    y_true = np.vstack(df["label_ids"].to_numpy()).astype(int)
    y_pred = to_indicator([[DEFAULT_LABEL]] * len(df))
    return compute_metrics(y_true, y_pred)


def evaluate(data, split_name: str, negation: bool = False) -> tuple[dict, pd.DataFrame]:
    df = load_split(data)

    start = time.perf_counter()
    predicted_labels = predict(df["text"], negation)
    inference_time = time.perf_counter() - start

    y_true = np.vstack(df["label_ids"].to_numpy()).astype(int)
    y_pred = to_indicator(predicted_labels)

    metrics = compute_metrics(y_true, y_pred)
    metrics["inference_time_seconds"] = inference_time
    metrics["inference_time_per_example_ms"] = (
        inference_time / max(len(df), 1) * 1000
    )
    metrics["labels_per_example"] = float(y_pred.sum(axis=1).mean())
    metrics["no_rule_fired"] = sum(
        1 for text in df["text"] if max(score_text(text, negation).values()) == 0
    )

    predictions_df = pd.DataFrame(
        {
            "text": df["text"],
            "true_label": df["label"].map(str),
            "predicted_label": [str(labels) for labels in predicted_labels],
        }
    )

    print(f"\n{split_name} ({len(df)} examples)")
    print(json.dumps({k: float(v) for k, v in metrics.items()}, indent=2))
    return metrics, predictions_df


def per_label_report(data, negation: bool = False) -> pd.DataFrame:
    """F1 per genre: shows which rules work and which ones do not."""
    df = load_split(data)
    y_true = np.vstack(df["label_ids"].to_numpy()).astype(int)
    y_pred = to_indicator(predict(df["text"], negation))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(len(LABELS)), zero_division=0
    )
    return pd.DataFrame(
        {
            "genre": LABELS,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )


def confusion_matrix(data, negation: bool = False) -> pd.DataFrame:
    """Confusion matrix on the single-label examples, as required by the course.

    Multi-label examples are left out because a row must sum to one example.
    """
    df = load_split(data)
    predicted = predict(df["text"], negation)
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    for true, pred in zip(df["label"], predicted):
        if len(true) == 1:
            matrix[LABEL2ID[true[0]], LABEL2ID[pred[0]]] += 1
    return pd.DataFrame(matrix, index=LABELS, columns=LABELS)


def save_results(output_dir: str | Path, name: str, metrics: dict, predictions_df) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    predictions_df.to_csv(output_path / f"{name}_predictions.csv", index=False)
    with open(output_path / f"{name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

    print(f"Predictions saved to {output_path / f'{name}_predictions.csv'}")


def save_config(output_dir: str | Path, validation_metrics: dict, negation: bool) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "method": "deterministic keyword rules (no machine learning)",
                "paradigm": "symbolic: hand-written lexicon and rules",
                "representation": "binary bag-of-words limited to the lexicon",
                "preprocessing": ["html unescape", "NFKC", "strip urls/tags",
                                  "collapse whitespace", "lowercase"],
                "negation_handling": negation,
                "strong_weight": STRONG_WEIGHT,
                "weak_weight": WEAK_WEIGHT,
                "extra_label_ratio": EXTRA_LABEL_RATIO,
                "default_label": DEFAULT_LABEL,
                "tie_break": TIE_BREAK,
                "labels": LABELS,
                "keywords_per_genre": {
                    genre: {
                        "strong": len(groups["strong"]),
                        "weak": len(groups["weak"]),
                    }
                    for genre, groups in RULES.items()
                },
                "validation_metrics": {
                    k: float(v) for k, v in validation_metrics.items()
                },
            },
            f,
            indent=2,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the TextBench rule-based baseline (no machine learning)."
    )
    parser.add_argument("--validation", default="data/processed/validation.csv")
    parser.add_argument("--test", default="data/processed/test.csv")
    parser.add_argument("--output", default="results/baseline")
    parser.add_argument(
        "--negation",
        action="store_true",
        help="Ignore keywords inside a negation scope. Measured, then left off.",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate on the test set. Rules must be frozen before this.",
    )
    args = parser.parse_args()

    total_terms = sum(
        len(groups["strong"]) + len(groups["weak"]) for groups in RULES.values()
    )
    print("Method: deterministic keyword rules (no training, no random seed needed)")
    print(f"Genres: {len(LABELS)}  Keywords: {total_terms}")
    print(f"Weights: strong={STRONG_WEIGHT}  weak={WEAK_WEIGHT}")
    print(f"Decode: best score + extra genres if score >= {EXTRA_LABEL_RATIO} x best")
    print(f"Fallback when no rule fires: {DEFAULT_LABEL}")
    print(f"Negation handling: {'on' if args.negation else 'off'}")

    reference = majority_reference(args.validation)
    print(
        f"\nReference (always '{DEFAULT_LABEL}'): "
        f"accuracy={reference['accuracy']:.3f}  "
        f"f1_macro={reference['f1_macro']:.3f}"
    )

    validation_metrics, validation_predictions = evaluate(
        args.validation, "Validation", args.negation
    )
    save_results(args.output, "validation", validation_metrics, validation_predictions)
    save_config(args.output, validation_metrics, args.negation)

    print("\nPer genre (validation):")
    print(per_label_report(args.validation, args.negation).to_string(index=False))

    if args.evaluate_test:
        print("\nFinal Test Evaluation...")
        test_metrics, test_predictions = evaluate(args.test, "Test", args.negation)
        save_results(args.output, "test", test_metrics, test_predictions)

        print("\nPer genre (test):")
        print(per_label_report(args.test, args.negation).to_string(index=False))

        print("\nConfusion matrix (test, single-label examples):")
        print(confusion_matrix(args.test, args.negation).to_string())


if __name__ == "__main__":
    main()
