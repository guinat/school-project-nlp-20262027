"""Comparison and error analysis of the three systems.

Reads only the artifacts written by ``run_evaluation.py --stage systems``
(scores, timing, costs) plus the delivered configs of the three systems, so it
runs in a few seconds without any model. Everything lands in
results/evaluation/: tables (CSV + Markdown), figures (PNG), erreurs.md and
key_numbers.json, the single source of every number quoted in the report.

Protocol reminders:
- the test set is only scored; every choice (decoding threshold) is made on
  validation, then applied once to test;
- the "tel que livré" rows are never modified; the harmonised decoding is an
  additional analysis, reported as extra rows.
"""

import json

import numpy as np
import pandas as pd

import evaluation as ev
from evaluation import LABELS
from run_evaluation import (
    LESS_ARTIFACT,
    OUT,
    RESULTS,
    THRESHOLD_GRID,
    baseline_decode,
    load_scores,
    read_json,
    split_data,
    write_json,
)

SMALL_CLASSES = ["psychology", "romance", "sports", "travel"]
NAMES = {
    "baseline": "Baseline",
    "deep_learning": "Deep Learning",
    "transformer": "Transformer",
}
N_BOOT = 1000


def grid_label(value) -> str:
    return "top-1 seul" if np.isinf(value) else f"{value:.1f}"


def data_numbers() -> dict:
    """Facts about the data, read from the processed files (and, for the raw
    size, from the saved output of notebooks/cleaning.ipynb)."""
    import re

    root = OUT.parents[1]
    notebook = json.loads((root / "notebooks" / "cleaning.ipynb").read_text(encoding="utf-8"))
    raw_rows = None
    for cell in notebook["cells"]:
        if "df.shape" in "".join(cell["source"]) and "Dimensions" in "".join(cell["source"]):
            for output in cell.get("outputs", []):
                text = "".join(output.get("data", {}).get("text/plain", ""))
                match = re.match(r"\((\d+), \d+\)", text)
                if match:
                    raw_rows = int(match.group(1))
    clean = pd.read_csv(root / "data" / "processed" / "data_clean.csv")
    splits = {s: split_data(s) for s in ["train", "validation", "test"]}
    words = pd.concat([d["text"] for d in splits.values()]).str.split().str.len()
    return {
        "raw_rows_cleaning_notebook": raw_rows,
        "clean_rows": int(len(clean)),
        "split_sizes": {s: int(len(d)) for s, d in splits.items()},
        "split_shares": {s: float(len(d) / len(clean)) for s, d in splits.items()},
        "multi_label": {s: int(sum(len(t) > 1 for t in d["labels"])) for s, d in splits.items()},
        "support": {s: {g: int(sum(g in t for t in d["labels"])) for g in LABELS}
                    for s, d in splits.items()},
        "words_median": float(words.median()),
        "words_min": int(words.min()),
        "words_max": int(words.max()),
        "with_less_artifact": {s: int(d["text"].str.contains(LESS_ARTIFACT).sum())
                               for s, d in splits.items()},
    }


# Predictions
def delivered_thresholds() -> dict:
    """Decoding parameters as delivered, read from each system's config."""
    return {
        "baseline": read_json(RESULTS / "baseline" / "config.json")["extra_label_ratio"],
        "deep_learning": read_json(RESULTS / "deep_learning" / "config.json")["extra_label_threshold"],
        "transformer": read_json(RESULTS / "transformer" / "config.json")["prediction_threshold"],
    }


def decoders() -> dict:
    """Harmonised rule for each system: top-1 + every genre above a threshold.

    For the networks the threshold applies to probabilities. The baseline has
    no probabilities: its threshold is the ratio to the best score, which is
    exactly its own rule (score >= ratio x best).
    """
    return {
        "baseline": baseline_decode,
        "deep_learning": ev.decode_top_threshold,
        "transformer": ev.decode_top_threshold,
    }


def delivered_predictions(split: str) -> dict:
    thresholds = delivered_thresholds()
    return {
        "baseline": baseline_decode(load_scores("baseline", split), thresholds["baseline"]),
        "deep_learning": ev.decode_top_threshold(
            load_scores("deep_learning", split), thresholds["deep_learning"]),
        "transformer": ev.decode_independent(
            load_scores("transformer", split), thresholds["transformer"]),
    }


def check_against_delivered_files(predictions: dict) -> None:
    """The recomputed test predictions must be the delivered ones."""
    for system, labels in predictions.items():
        delivered = pd.read_csv(RESULTS / system / "test_predictions.csv")
        delivered = delivered["predicted_label"].map(ev.parse_labels)
        same = [set(a) == set(b) for a, b in zip(labels, delivered)]
        if not all(same):
            raise AssertionError(f"{system}: {len(same) - sum(same)} predictions differ")


def select_decoding(validation_truth) -> tuple[dict, pd.DataFrame]:
    """Choose the threshold of each system on validation only."""
    chosen, tables = {}, []
    for system, decode in decoders().items():
        scores = load_scores(system, "validation")
        best, table = ev.select_threshold(
            validation_truth, lambda t: decode(scores, t), THRESHOLD_GRID)
        chosen[system] = float(best)
        table.insert(0, "système", NAMES[system])
        tables.append(table)
    grid = pd.concat(tables, ignore_index=True)
    grid["threshold"] = grid["threshold"].map(grid_label)
    return chosen, grid


# Tables
def train_seconds() -> dict:
    retrain = read_json(OUT / "dl_reproducibility.json")
    return {
        "baseline": 0.0,
        "deep_learning": float(np.median([run["train_seconds"] for run in retrain])),
        "transformer": float(read_json(RESULTS / "transformer" / "config.json")["train_time_seconds"]),
    }


def inference_ms(device: str = "cpu") -> dict:
    timing = read_json(OUT / "timing.json")["timings"]
    return {t["system"]: t["ms_per_example"] for t in timing if t["device"] == device}


def build_rows(truth, delivered: dict, harmonised: dict, chosen: dict,
               extra_rows: list) -> list:
    train, infer = train_seconds(), inference_ms("cpu")
    rows = [
        {"key": system, "name": NAMES[system], "metrics": ev.evaluate(truth, preds),
         "train_seconds": train[system], "inference_ms": infer[system],
         "predictions": preds}
        for system, preds in delivered.items()
    ]
    for system in extra_rows:
        rows.append({
            "key": f"{system}_harmonise",
            "name": f"{NAMES[system]}, décodage harmonisé (seuil {grid_label(chosen[system])})",
            "metrics": ev.evaluate(truth, harmonised[system]),
            "train_seconds": train[system], "inference_ms": infer[system],
            "predictions": harmonised[system],
        })
    return rows


def detail_table(rows: list) -> pd.DataFrame:
    return pd.DataFrame([
        {"Méthode": row["name"], **{k: v for k, v in row["metrics"].items()}}
        for row in rows
    ])


# Error analysis
def correctness(truth, predictions) -> np.ndarray:
    return (ev.to_indicator(truth) == ev.to_indicator(predictions)).all(axis=1)


def error_candidates(truth, preds: dict, bets: dict, probs: dict) -> dict:
    """Reproducible criteria, fixed before looking at the examples.

    Each criterion is a boolean mask over the test set. ``bets`` are the top-1
    genres (from scores), ``probs`` the probabilities of the two networks.
    """
    single = np.array([len(t) == 1 for t in truth])
    true1 = np.array([t[0] if len(t) == 1 else None for t in truth], dtype=object)
    right = {s: correctness(truth, p) for s, p in preds.items()}
    bet = {s: np.array(b, dtype=object) for s, b in bets.items()}
    net_bets = [bet["deep_learning"], bet["transformer"]]

    trio = {"thriller", "crime", "horror"}
    in_trio = np.array([t in trio for t in true1])
    trio_confusion = in_trio & np.any(
        [np.array([b in trio and b != t for b, t in zip(b_, true1)]) for b_ in net_bets], axis=0)

    fantasy_horror = np.any([
        np.array([(t, b) in {("fantasy", "horror"), ("horror", "fantasy")}
                  for t, b in zip(true1, b_)]) for b_ in net_bets], axis=0)

    # Both networks bet on the same genre against the label, both with p >= 0.8.
    net_argmax = [probs[s].argmax(axis=1) for s in ["deep_learning", "transformer"]]
    doubtful = single & np.array([
        a == b and LABELS[a] != t
        and probs["deep_learning"][i, a] >= 0.8 and probs["transformer"][i, b] >= 0.8
        for i, (a, b, t) in enumerate(zip(*net_argmax, true1))])

    return {
        "A. les trois systèmes se trompent": ~right["baseline"] & ~right["deep_learning"] & ~right["transformer"],
        "B. seule la baseline se trompe": ~right["baseline"] & right["deep_learning"] & right["transformer"],
        "C. le transformer ne prédit aucun genre": np.array([len(p) == 0 for p in preds["transformer"]]),
        "D. confusion thriller / crime / horror (DL ou transformer)": trio_confusion,
        "E. confusion fantasy / horror (DL ou transformer)": fantasy_horror,
        "F. exemple à plusieurs genres": ~single,
        "G. genre douteux : DL et transformer d'accord contre l'étiquette, tous deux à p >= 0.8": doubtful,
        "H. petite classe manquée par le DL": single & np.isin(true1, SMALL_CLASSES) & ~right["deep_learning"],
    }


PICKS = {  # how many examples to draw in each criterion
    "A": 2, "B": 1, "C": 1, "D": 1, "E": 1, "F": 1, "G": 1, "H": 1,
}


def select_errors(candidates: dict, seed: int = ev.SEED) -> list:
    """Draw examples in each criterion with a fixed seed, criteria in fixed
    order, never the same example twice. No hand-picking."""
    rng = np.random.default_rng(seed)
    taken, selection = set(), []
    for criterion, mask in candidates.items():
        pool = [int(i) for i in np.flatnonzero(mask) if int(i) not in taken]
        k = min(PICKS[criterion[0]], len(pool))
        for index in sorted(rng.choice(pool, size=k, replace=False).tolist()) if k else []:
            taken.add(index)
            selection.append((criterion, index))
    return selection


# Hypotheses written after reading each selected example (see erreurs.md).
# Keyed by test index; the selection itself is automatic.
ERROR_NOTES: dict[int, str] = {
    38: "Résumé de *Ring* (K. Suzuki) écrit comme une enquête (reporter, "
        "« investigation », « witnesses ») : l'horreur n'est portée que par la "
        "cassette maudite, sans mot-clé explicite (ghost, haunted). La baseline "
        "tombe dans le piège lexical (crime). Le transformer place bien horror en "
        "tête (0,49) mais juste sous le seuil de 0,5 : il ne prédit rien.",
    379: "Résumé de 23 mots sans aucun indice de genre (*La Peau froide*, roman "
         "d'horreur) : aucune méthode fondée sur le texte seul ne peut trouver "
         "horror. La baseline s'accroche à « island » (travel). Limite de la "
         "donnée, pas du modèle.",
    236: "Piège lexical : le vocabulaire de l'enquête (murderer, investigation, "
         "police) donne 4,5 points à crime contre 1 à thriller (« killer », mot "
         "faible). Ce qui fait de ce livre un thriller (série Jack Reacher, "
         "action) n'est pas dans le lexique. Le DL et le transformer ont appris "
         "cette frontière sur les données et ont raison.",
    219: "Le suspense est suggéré (« terrible legacy », « either the girls run, "
         "or they die », « darkness ») mais jamais nommé. Les indices vont autant "
         "vers horror que vers romance : le DL hésite (0,48 / 0,48) et le "
         "transformer ne dépasse 0,5 pour aucun genre. Frontière floue entre "
         "thriller psychologique et horreur.",
    436: "*The Girl Next Door* (J. Ketchum) est souvent classé en horreur : la "
         "prédiction horror du DL et du transformer est défendable et l'étiquette "
         "thriller discutable. La baseline est piégée par « a quick tour of his "
         "neighborhood » et « travel » : le lexique ignore le contexte.",
    116: "*Firmin* (S. Savage), fable littéraire sur un rat qui apprend à lire. "
         "Aucun mot du lexique : la baseline renvoie le genre par défaut. Le "
         "fantastique vient d'un animal qui lit, ce qui demande de comprendre le "
         "sens ; « rat », « alcoholic », « survive » tirent les deux réseaux vers "
         "horror. L'étiquette fantasy est elle-même discutable.",
    481: "*City of Bones* (C. Clare), étiqueté romance + fantasy. Les deux réseaux "
         "trouvent fantasy mais pas romance, seulement suggérée par un personnage "
         "(« looks a little like an angel and acts a lot like a jerk »). La "
         "baseline est piégée par le vocabulaire du meurtre (crime 7 points). Les "
         "exemples multi-genres sont rares à l'entraînement : les modèles "
         "prédisent surtout un seul genre.",
    136: "*Neon Gods* (K. Robert), réécriture de Hadès et Perséphone : une "
         "romance dans un décor mythologique. Les trois systèmes prédisent "
         "fantasy, les réseaux avec confiance (0,85 et 0,94). L'étiquette unique "
         "romance est incomplète plutôt que fausse : romance + fantasy aurait été "
         "plus juste. Erreur de la donnée plus que du modèle.",
    12: "*Perfect Chemistry* (S. Elkeles) : la baseline a raison (passionate, "
        "boyfriend, chemistry). Le DL préfère thriller (gang, secret) ; le "
        "transformer place romance en tête (0,38) mais sous le seuil de 0,5. "
        "romance est une petite classe (88 résumés d'entraînement sur 3 580) : "
        "les réseaux y sont peu confiants.",
}


def fmt_probs(row, k: int = 3) -> str:
    order = np.argsort(-row)[:k]
    return ", ".join(f"{LABELS[j]} {row[j]:.2f}" for j in order)


def explain_baseline(text: str) -> str:
    import baseline as bl

    info = bl.explain(text, top_k=3)
    if info["top_scores"][0][1] == 0:
        return f"aucun mot-clé, genre par défaut ({bl.DEFAULT_LABEL})"
    parts = []
    for genre, score in info["top_scores"]:
        kw = info["matched_keywords"][genre]
        words = kw["strong"] + kw["weak"]
        if score:
            parts.append(f"{genre} {score:g} ({', '.join(words)})")
    return "; ".join(parts)


def error_records(selection, test, preds, probs, harmonised) -> pd.DataFrame:
    records = []
    for criterion, i in selection:
        text = test["text"].iloc[i]
        records.append({
            "critère": criterion,
            "index": i,
            "extrait": text[:200].replace("\n", " ") + ("…" if len(text) > 200 else ""),
            "n_mots": len(text.split()),
            "vrai": ", ".join(test["labels"].iloc[i]),
            "baseline": ", ".join(preds["baseline"][i]) or "(aucun)",
            "deep_learning": ", ".join(preds["deep_learning"][i]) or "(aucun)",
            "transformer": ", ".join(preds["transformer"][i]) or "(aucun)",
            "transformer_harmonise": ", ".join(harmonised[i]),
            "proba_dl": fmt_probs(probs["deep_learning"][i]),
            "proba_transformer": fmt_probs(probs["transformer"][i]),
            "baseline_explain": explain_baseline(text),
            "hypothèse": ERROR_NOTES.get(i, "à commenter"),
        })
    return pd.DataFrame(records)


def errors_markdown(records: pd.DataFrame, candidates: dict) -> str:
    lines = [
        "# Analyse d'erreurs (test, prédictions telles que livrées)",
        "",
        "Sélection automatique : pour chaque critère (ordre fixe), tirage aléatoire "
        f"avec `numpy.random.default_rng({ev.SEED})` parmi les exemples qui le "
        "vérifient, sans remise entre critères. Aucun exemple choisi à la main.",
        "",
        "| Critère | Candidats | Retenus |",
        "| --- | --- | --- |",
    ]
    for criterion, mask in candidates.items():
        kept = int((records["critère"] == criterion).sum())
        lines.append(f"| {criterion} | {int(mask.sum())} | {kept} |")
    lines.append("")
    for n, row in enumerate(records.itertuples(index=False), start=1):
        lines += [
            f"## Erreur {n} — test #{row.index} ({row.critère})",
            "",
            f"> {row.extrait}",
            "",
            f"- Longueur : {row.n_mots} mots",
            f"- **Vrai genre** : {row.vrai}",
            f"- Baseline : {row.baseline} — mots-clés : {row.baseline_explain}",
            f"- Deep Learning : {row.deep_learning} — probabilités : {row.proba_dl}",
            f"- Transformer : {row.transformer} — probabilités : {row.proba_transformer}",
            f"- Transformer, décodage harmonisé : {row.transformer_harmonise}",
            f"- **Hypothèse** : {row.hypothèse}",
            "",
        ]
    return "\n".join(lines)


# Main
def run_analysis() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    validation, test = split_data("validation"), split_data("test")
    probs = {s: load_scores(s, "test") for s in ["deep_learning", "transformer"]}
    baseline_test_scores = load_scores("baseline", "test")
    score_of = {"baseline": baseline_test_scores, **probs}

    delivered = {split: delivered_predictions(split) for split in ["validation", "test"]}
    check_against_delivered_files(delivered["test"])

    # 1. Harmonised decoding: threshold chosen on validation, applied once to test.
    chosen, grid = select_decoding(validation["labels"])
    ev.save_table(grid, OUT / "decodage_grille_validation")
    harmonised = {
        split: {s: decoders()[s](load_scores(s, split), chosen[s]) for s in chosen}
        for split in ["validation", "test"]
    }
    thresholds = delivered_thresholds()
    # A harmonised row is shown only if it changes the delivered decoding.
    extra = [s for s in chosen
             if s == "transformer" or not np.isclose(chosen[s], thresholds[s])]

    # 2. Final tables, validation and test.
    rows = {}
    for split, data in [("validation", validation), ("test", test)]:
        rows[split] = build_rows(data["labels"], delivered[split], harmonised[split],
                                 chosen, extra)
        table = ev.comparison_table(rows[split])
        ev.save_table(table, OUT / f"tableau_final_{split}")
        ev.save_table(ev.format_table(table), OUT / f"tableau_final_{split}_lisible")
        ev.save_table(detail_table(rows[split]), OUT / f"metriques_detail_{split}")

    # 3. Timing and cost table.
    timing = read_json(OUT / "timing.json")
    costs = read_json(OUT / "costs.json")
    train = train_seconds()
    cost_rows = []
    for system in NAMES:
        by_device = {t["device"]: t for t in timing["timings"] if t["system"] == system}
        cost_rows.append({
            "Méthode": NAMES[system],
            "Paramètres appris": costs[system]["learned_parameters"],
            "Taille disque (Mo)": costs[system]["disk_bytes"] / 1e6,
            "Train (s)": train[system],
            "Inférence CPU (ms/ex.)": by_device["cpu"]["ms_per_example"],
            "Inférence MPS (ms/ex.)": by_device.get("mps", {}).get("ms_per_example", np.nan),
            "Test complet CPU (s)": by_device["cpu"]["total_seconds_median"],
        })
    ev.save_table(pd.DataFrame(cost_rows), OUT / "couts_temps")

    # 4. Per-class F1 and one-vs-rest counts.
    per_class = {row["key"]: ev.per_class_report(test["labels"], row["predictions"])
                 for row in rows["test"]}
    wide = per_class["baseline"][["genre", "support"]].copy()
    for row in rows["test"]:
        wide[row["name"]] = per_class[row["key"]]["f1"].to_numpy()
    ev.save_table(wide, OUT / "f1_par_classe_test")
    long = pd.concat([df.assign(système=key) for key, df in per_class.items()])
    ev.save_table(long, OUT / "rapport_par_classe_test")
    plot_keys = ["baseline", "deep_learning", "transformer", "transformer_harmonise"]
    ev.plot_per_class_f1(
        {("Transformer harmonisé" if k.endswith("harmonise") else NAMES[k]): per_class[k]
         for k in plot_keys},
        OUT / "f1_par_classe_test.png")

    # 5. Confusion matrices (single-label test examples, top-1 from scores).
    matrices = {}
    for system in NAMES:
        m = ev.confusion_matrix(test["labels"], delivered["test"][system], score_of[system])
        m.to_csv(OUT / f"confusion_{system}.csv")
        matrices[NAMES[system]] = m
    ev.plot_confusion_matrices(matrices, OUT / "matrices_confusion.png")
    m_h = ev.confusion_matrix(test["labels"], harmonised["test"]["transformer"], probs["transformer"])
    m_h.to_csv(OUT / "confusion_transformer_harmonise.csv")
    ev.plot_confusion_matrices(
        {"Transformer (tel que livré)": matrices["Transformer"],
         f"Transformer, décodage harmonisé (seuil {grid_label(chosen['transformer'])})": m_h},
        OUT / "matrices_confusion_transformer_harmonise.png")

    # 6. Loss curves.
    dl_config = read_json(RESULTS / "deep_learning" / "config.json")
    dl_best = int(np.argmax([m["f1_macro"] for m in dl_config["validation_metrics"]]) + 1)
    last_checkpoint = max((RESULTS / "transformer").glob("checkpoint-*"),
                          key=lambda p: int(p.name.split("-")[1]))
    state = read_json(last_checkpoint / "trainer_state.json")
    train_log = [e for e in state["log_history"] if "loss" in e]
    eval_log = [e for e in state["log_history"] if "eval_loss" in e]
    steps_per_epoch = state["max_steps"] / state["num_train_epochs"]
    tr_best = int(round(state["best_global_step"] / steps_per_epoch))
    curves = {
        "Deep Learning — BCE pondérée (pos_weight)": {
            "epochs": list(range(1, len(dl_config["train_loss"]) + 1)),
            "train": dl_config["train_loss"],
            "validation": dl_config["validation_loss"],
            "best_epoch": dl_best,
        },
        "Transformer — BCE": {
            "epochs": [e["epoch"] for e in eval_log],
            "train": [e["loss"] for e in train_log],
            "validation": [e["eval_loss"] for e in eval_log],
            "best_epoch": tr_best,
        },
    }
    ev.plot_loss_curves(curves, OUT / "courbes_loss.png")
    write_json(curves, OUT / "courbes_loss.json")

    # 7. Robustness: bootstrap CI and McNemar.
    robustness = {"bootstrap_f1_macro_test": {}, "paired": {}}
    for row in rows["test"]:
        robustness["bootstrap_f1_macro_test"][row["name"]] = ev.bootstrap_f1_macro(
            test["labels"], row["predictions"], n_boot=N_BOOT)
    by_key = {row["key"]: row["predictions"] for row in rows["test"]}
    for a, b in [("deep_learning", "transformer"),
                 ("deep_learning", "transformer_harmonise"),
                 ("transformer_harmonise", "transformer"),
                 ("deep_learning", "baseline")]:
        robustness["paired"][f"{a} vs {b}"] = {
            "f1_macro_difference": ev.paired_bootstrap_difference(
                test["labels"], by_key[a], by_key[b], n_boot=N_BOOT),
            "mcnemar_exact_match": ev.mcnemar_test(test["labels"], by_key[a], by_key[b]),
        }
    write_json(robustness, OUT / "robustesse.json")
    boot = pd.DataFrame([
        {"Méthode": name, "F1 macro": r["f1_macro"], "IC 95 % bas": r["ci_low"],
         "IC 95 % haut": r["ci_high"]}
        for name, r in robustness["bootstrap_f1_macro_test"].items()])
    ev.save_table(boot, OUT / "bootstrap_f1_macro_test")

    # 8. Error analysis.
    bets = {s: ev.top1(delivered["test"][s], score_of[s]) for s in NAMES}
    candidates = error_candidates(test["labels"], delivered["test"], bets, probs)
    selection = select_errors(candidates)
    records = error_records(selection, test, delivered["test"], probs,
                            harmonised["test"]["transformer"])
    records.to_csv(OUT / "erreurs.csv", index=False)
    (OUT / "erreurs.md").write_text(errors_markdown(records, candidates), encoding="utf-8")

    # 9. Scraping artifact "(less)": share per genre (train) and effect on the
    # transformer when it is removed from the test summaries.
    train = split_data("train")
    has_less = train["text"].str.contains(LESS_ARTIFACT)
    less_share = {g: float(has_less[[g in t for t in train["labels"]]].mean()) for g in LABELS}
    cleaned = load_scores("transformer", "test_sans_less")
    artifact = {
        "train_share_with_less_by_genre": less_share,
        "test_with_less": int(test["text"].str.contains(LESS_ARTIFACT).sum()),
        "transformer_delivered": ev.evaluate(test["labels"], delivered["test"]["transformer"]),
        "transformer_delivered_without_less": ev.evaluate(test["labels"], ev.decode_independent(
            cleaned, thresholds["transformer"])),
        "transformer_harmonised": ev.evaluate(test["labels"], harmonised["test"]["transformer"]),
        "transformer_harmonised_without_less": ev.evaluate(test["labels"], ev.decode_top_threshold(
            cleaned, chosen["transformer"])),
        "transformer_changed_predictions": int(sum(
            set(a) != set(b) for a, b in zip(delivered["test"]["transformer"],
                                             ev.decode_independent(cleaned, thresholds["transformer"])))),
    }
    write_json(artifact, OUT / "artefact_less.json")

    # 10. Key numbers quoted in the report.
    empty = np.array([len(p) == 0 for p in delivered["test"]["transformer"]])
    argmax_right = np.array([
        LABELS[int(np.argmax(probs["transformer"][i]))] in test["labels"].iloc[i]
        for i in range(len(test))])
    right = {s: correctness(test["labels"], p) for s, p in delivered["test"].items()}
    f1_small = {row["name"]: float(per_class[row["key"]].set_index("genre").loc[SMALL_CLASSES, "f1"].mean())
                for row in rows["test"]}
    f1_large = {row["name"]: float(per_class[row["key"]].set_index("genre").drop(SMALL_CLASSES)["f1"].mean())
                for row in rows["test"]}
    checks = read_json(OUT / "checks.json")
    key = {
        "data": data_numbers(),
        "test_size": len(test),
        "validation_size": len(validation),
        "train_size": len(split_data("train")),
        "test_multi_label": int(sum(len(t) > 1 for t in test["labels"])),
        "test_support": {g: int(sum(g in t for t in test["labels"])) for g in LABELS},
        "decoding_delivered": thresholds,
        "decoding_chosen_on_validation": {s: grid_label(v) for s, v in chosen.items()},
        "transformer_empty_test": int(empty.sum()),
        "transformer_empty_test_argmax_right": int((empty & argmax_right).sum()),
        "transformer_empty_test_dl_right": int((empty & right["deep_learning"]).sum()),
        "transformer_empty_test_max_proba_median": float(np.median(probs["transformer"][empty].max(axis=1))),
        "baseline_multi_predictions_test": int(sum(len(p) > 1 for p in delivered["test"]["baseline"])),
        "deep_learning_multi_predictions_test": int(sum(len(p) > 1 for p in delivered["test"]["deep_learning"])),
        "all_three_wrong_test": int((~right["baseline"] & ~right["deep_learning"] & ~right["transformer"]).sum()),
        "all_three_right_test": int((right["baseline"] & right["deep_learning"] & right["transformer"]).sum()),
        "f1_mean_small_classes": f1_small,
        "f1_mean_large_classes": f1_large,
        "baseline_no_rule_fired_test": checks["baseline_test_no_rule_fired"],
        "transformer_test_truncated_share": checks["transformer_test_truncated_share"],
        "transformer_test_median_tokens": checks["transformer_test_median_tokens"],
        "error_candidates": {c: int(m.sum()) for c, m in candidates.items()},
        "artifact_less": {k: v for k, v in artifact.items()},
        "checks": checks,
    }
    write_json(key, OUT / "key_numbers.json")

    print(ev.to_markdown(ev.format_table(ev.comparison_table(rows["test"]))))
    return key


if __name__ == "__main__":
    print(json.dumps(run_analysis(), indent=2, ensure_ascii=False, default=float))
