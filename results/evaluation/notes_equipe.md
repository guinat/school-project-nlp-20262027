# Notes pour l'équipe (évaluation, Arno)

Aucun fichier des autres membres n'a été modifié. Les points ci-dessous sont des
constats à discuter. Chaque chiffre vient de `results/evaluation/`.

## Données (Elyes)

1. **Artefact de scraping « (less) »** : c'est le bouton « show less » de Goodreads, resté dans
   910 résumés d'entraînement sur 3 580 et dans 131 résumés de test sur 508. Sa présence dépend
   du genre. En train, on le trouve dans 100 % des romances, 99 % des sports, 50 % des thrillers
   et 43 % des fantasy, mais dans 0 % des crime, history, psychology et travel
   (`artefact_less.json`). C'est un raccourci potentiel.
   Effet mesuré : il est nul pour la baseline (« less » n'est pas dans le lexique) et pour le DL
   (« less » est un stop word de sklearn). Il est faible pour le transformer : sans l'artefact,
   15 prédictions de test changent et le F1 macro passe de 0,654 à 0,649.
   Proposition : le retirer dans `cleaning.ipynb`.
2. **Étiquettes discutables** repérées dans l'analyse d'erreurs (`erreurs.md`) :
   - test #436, *The Girl Next Door* (J. Ketchum) : étiqueté thriller, souvent classé en horreur ;
   - test #136, *Neon Gods* : étiqueté romance seule, alors que c'est aussi de la fantasy ;
   - test #116, *Firmin* : étiqueté fantasy, alors que c'est plutôt de la fiction littéraire.
3. **Résumés très courts** : 6 mots au minimum sur les trois splits, 10 en test. Le test #379 (23 mots) ne contient
   aucun indice de genre.
4. Aucun doublon, aucun chevauchement entre train, validation et test (vérifié sur les textes normalisés).
5. **Seed du découpage : 67** (`FIXED_SEED = 67`, `random_state=67` dans `spliting.ipynb`), alors que
   les modèles et l'évaluation utilisent 42. Ce n'est pas faux (le découpage est figé et versionné),
   mais il faut l'écrire dans le README et le rapport. Refaire le découpage avec 42 invaliderait
   tous les résultats.
6. **Dépendance manquante** : `spliting.ipynb` importe `iterstrat`, fourni par le paquet
   `iterative-stratification`, qui n'est pas dans `pyproject.toml`. Le notebook ne tourne pas
   après un simple `uv sync`.
7. **Provenance non documentée** : `data/raw/data.csv` (4 657 lignes d'après `cleaning.ipynb`)
   n'est pas versionné et sa source (URL, licence) n'est écrite nulle part. D'après les
   artefacts présents dans les textes (« (less) », « ==Reception »), les résumés viennent de
   Goodreads et de Wikipédia. Il faut ajouter la source exacte au README (checklist 13.1).
8. Le test représente 11,2 % du corpus nettoyé et non 10 % : la stratification multi-label
   itérative n'atteint pas exactement les proportions demandées. C'est sans conséquence, mais
   à savoir.

## Baseline (Clément)

- Pas de bug. Relancée sur la validation et le test : les prédictions sont identiques, ordre compris.
- `extra_label_ratio = 0.9` : en test, 78 prédictions sur 508 ont plusieurs genres, contre 6 vrais
  multi-genres. Sur la validation, « top-1 seul » donne un F1 macro de 0,487 contre 0,486 avec 0,9 :
  c'est une égalité, le choix de 0,9 est défendable.

## Deep Learning (Faiki)

- `predict()` renvoie les genres dans l'ordre alphabétique, pas par score : le premier genre n'est
  pas forcément le top-1. Ce n'est pas bloquant, l'évaluation commune recalcule le top-1 à partir
  des probabilités.
- Attention : lancé sans `--evaluate-test`, `python src/deep_learning.py` **écrase**
  `results/deep_learning/best_model.pt`. Pour la vérification de reproductibilité, j'ai réentraîné
  en mémoire, sans rien écraser.
- Reproductibilité : deux réentraînements avec la seed 42 sur MPS donnent des prédictions de test
  identiques à 100 % à celles livrées (écart de probabilité maximal de 3e-7,
  `dl_reproducibility.json`).

## Transformer (Nathan)

- `transformer.py` n'expose pas de fonction `predict(texts) -> list[list[str]]` : l'inférence
  passe par `Trainer.predict`. J'en ai écrit une dans `src/run_evaluation.py`
  (`transformer_probabilities`), sans toucher à `transformer.py`.
- Le seuil indépendant de 0,5 laisse **83 prédictions vides sur 508** en test. Pour 41 d'entre
  elles, le genre le plus probable était le bon.
- La classe romance n'est jamais prédite en test (F1 = 0) : la loss n'a pas de pondération de
  classe, contrairement au DL (`pos_weight`).
- Troncature à 256 tokens : 48 % des résumés du test sont coupés (médiane : 245 tokens).
- L'inférence recalculée en CPU fp32 donne exactement les mêmes prédictions que l'inférence
  livrée (MPS, bf16).

## Protocole

- Les temps d'inférence de chaque `test_metrics.json` ne sont pas comparables entre eux (device,
  batch et préchauffage différents). Il faut utiliser `timing.json` (mêmes conditions).
- Les métriques des trois `test_metrics.json` sont retrouvées exactement (écart de 0) par
  `src/evaluation.py`.
