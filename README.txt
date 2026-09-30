TextBench : prédire le genre d'un livre à partir de son résumé
==============================================================

Mini-projet NLP. Nous comparons trois approches de classification de texte sur les
mêmes données et avec le même protocole d'évaluation :
  1. une baseline sans apprentissage : un lexique de mots-clés et des règles ;
  2. un modèle de Deep Learning entraîné avec PyTorch : TF-IDF suivi d'un MLP ;
  3. un Transformer pré-entraîné et affiné : DistilBERT.

Équipe :
  - Elyes : données et protocole ;
  - Clément : baseline ;
  - Faiki : deep learning ;
  - Nathan : transformer ;
  - Arno : évaluation et analyse.

Livrables : notebooks/projet.ipynb (exécuté), rapport.pdf (4 pages + annexes) et ce fichier.


1. Cas d'usage
--------------
Une librairie en ligne ou une bibliothèque doit ranger des fiches de livres dont le genre
manque. À partir du seul résumé (en anglais), on propose un ou plusieurs genres parmi 10 :
crime, fantasy, history, horror, psychology, romance, science, sports, thriller, travel.

La tâche est multi-label, mais environ 98 % des résumés n'ont qu'un seul genre. La métrique
principale est le F1 macro, où chaque genre pèse autant : le corpus est déséquilibré, et
psychology, romance, sports et travel sont de petites classes.

Résultats sur le test (508 résumés), recopiés de results/evaluation/tableau_final_test_lisible.md :

  Méthode         Accuracy  Précision  Rappel  F1 macro  Train    Inférence (CPU)
  Baseline         0,415     0,426     0,613    0,464     0 s      1,15 ms/exemple
  Deep Learning    0,701     0,764     0,707    0,730     2,9 s    0,12 ms/exemple
  Transformer      0,650     0,754     0,584    0,654   451,5 s   10,27 ms/exemple

Système retenu : le Deep Learning. La justification est dans rapport.pdf, section 5.


2. Structure du dépôt
---------------------
  data/processed/          données nettoyées et découpées (versionnées)
    data_clean.csv         4 535 résumés uniques
    train.csv              3 580 résumés, colonnes : summary, genre
    validation.csv           447 résumés
    test.csv                 508 résumés (évaluation finale uniquement)
  notebooks/
    cleaning.ipynb         nettoyage du fichier brut (Elyes)
    spliting.ipynb         découpage train/validation/test (Elyes)
    projet.ipynb           notebook final, 14 sections, exécuté
  src/
    baseline.py            baseline à règles (Clément)
    deep_learning.py       TF-IDF + MLP PyTorch (Faiki)
    transformer.py         DistilBERT (Nathan)
    evaluation.py          protocole d'évaluation commun (Arno) : evaluate(),
                           confusion_matrix(), bootstrap, McNemar, chronométrage
    run_evaluation.py      exécute les 3 systèmes, sauvegarde leurs scores, mesure
                           les temps et vérifie la reproductibilité
    analysis.py            tableaux, figures et analyse d'erreurs
  results/                 sorties des systèmes (modèles non versionnés)
    evaluation/            tableaux (CSV et Markdown), figures (PNG), scores sauvegardés,
                           erreurs.md, notes_equipe.md, key_numbers.json (versionné)
  report/latex/            source LaTeX du rapport : rapport.tex et images/ (figures
                           copiées depuis results/evaluation/)
  rapport.pdf              rapport final
  pyproject.toml, uv.lock  dépendances


3. Version de Python et installation
------------------------------------
Python 3.12 (voir .python-version). Les dépendances sont gérées avec uv
(https://docs.astral.sh/uv/). Le projet a été testé avec uv 0.11.

  uv sync

Cette commande crée .venv et installe exactement les versions de uv.lock : torch,
transformers, scikit-learn, pandas, matplotlib, jupyter (nbconvert, ipykernel), etc.

Matériel : tout fonctionne sur CPU. Sur un Mac Apple Silicon, PyTorch utilise
automatiquement le GPU (MPS). Avec une carte NVIDIA, il utilise CUDA.


4. Données
----------
Les données prêtes à l'emploi sont versionnées dans data/processed/ : il n'y a rien à
télécharger pour exécuter le projet.

Le fichier brut data/raw/data.csv (4 657 résumés ; colonnes index, title, genre, summary)
n'est pas versionné. Il provient du jeu de données public « Book Genre Prediction »
publié sur Kaggle (résumés issus de Goodreads et de Wikipédia) :
  https://www.kaggle.com/datasets/athu1105/book-genre-prediction
La licence est indiquée sur cette page.

Pour refaire le nettoyage et le découpage :
  1. placer le fichier brut dans data/raw/data.csv ;
  2. exécuter notebooks/cleaning.ipynb, puis notebooks/spliting.ipynb.
spliting.ipynb a besoin en plus du paquet iterative-stratification :
  uv add iterative-stratification


5. Exécution
------------
Toutes les commandes se lancent depuis la racine du dépôt.

a) Notebook final (recommandé)
     uv run jupyter nbconvert --to notebook --execute --inplace notebooks/projet.ipynb
   Durée : environ 20 s. Le notebook importe les modules de src/, recharge les artefacts
   de results/, réentraîne le DL en mémoire (environ 3 s) et régénère tous les tableaux
   et figures de results/evaluation/. Trois drapeaux se règlent dans la section 2 :
     RETRAIN_DL = True             réentraîne le MLP, sans écraser le modèle livré
     RETRAIN_TRANSFORMER = False   True : réentraîne DistilBERT dans
                                   results/transformer_retrain (environ 7,5 min sur MPS)
     RUN_SYSTEMS = False           True : relance src/run_evaluation.py (environ 1 min)
   Le notebook peut aussi être ouvert dans Jupyter ou VS Code, avec le noyau .venv.

b) Chaque système séparément (réentraînement complet)
   Baseline, sans entraînement, en quelques secondes :
     uv run python src/baseline.py --evaluate-test
   Deep Learning, environ 3 à 5 s sur MPS. Attention, cette commande écrase
   results/deep_learning/best_model.pt :
     uv run python src/deep_learning.py
     uv run python src/deep_learning.py --evaluate-test
   Transformer, environ 7,5 min sur MPS en bf16. Le premier lancement télécharge
   distilbert-base-uncased depuis le Hugging Face Hub (environ 270 Mo) :
     uv run python src/transformer.py
     uv run python src/transformer.py --evaluate-test
   Chaque script écrit dans results/<système>/ : config.json, test_metrics.json et
   test_predictions.csv.

c) Évaluation commune
     uv run python src/run_evaluation.py     # systèmes, scores, temps (environ 1 min)
     uv run python src/analysis.py           # tableaux, figures, erreurs (environ 20 s)
   run_evaluation.py a besoin des modèles sauvegardés : results/deep_learning/best_model.pt
   et results/transformer/best_model/. Il ne réentraîne pas le transformer.
   analysis.py ne lit que results/evaluation/.
   Historique du Deep Learning (environ 10 s, nécessite git) :
     uv run python src/run_evaluation.py --stage history
   Cette étape réentraîne en mémoire la première version du modèle (commit 5cd4b7b, un genre
   dès que sa probabilité atteint 0,5) et écrit results/evaluation/dl_premiere_version.json.
   Elle montre pourquoi la règle de réponse a été changée ensuite.

d) Rapport
   La source est report/latex/rapport.tex. Pour la compiler, deux possibilités :
     - Overleaf : importer le dossier report/latex/ (compilateur pdfLaTeX) ;
     - en local : cd report/latex && tectonic rapport.tex (ou pdflatex, lancé deux fois).
   Copier ensuite report/latex/rapport.pdf à la racine du dépôt. Le corps du rapport
   (de l'introduction à la conclusion) doit rester sur 4 pages au plus. Tous les
   chiffres viennent des fichiers de results/evaluation/.


6. Les trois approches
----------------------
Baseline (src/baseline.py)
  Lexique de 272 mots-clés, écrits à la main sous forme d'expressions régulières,
  forts (poids 2,5) ou faibles (poids 1). Le système prédit le genre de meilleur score,
  plus tout genre dont le score atteint au moins 0,9 × le meilleur. Si aucune règle ne
  se déclenche, il prédit thriller. La fonction explain(texte) montre les mots-clés
  déclenchés. Il n'y a pas d'entraînement.

Deep Learning (src/deep_learning.py)
  TF-IDF (unigrammes et bigrammes, 20 000 features) suivi d'un MLP à une couche cachée
  de 256 unités, avec ReLU et dropout de 0,3. La loss est une BCE pondérée par classe,
  l'optimiseur Adam (lr 1e-3), le batch de 64. L'arrêt anticipé porte sur le F1 macro de
  validation (patience 4). Décodage : le genre le plus probable, plus tout genre de
  probabilité ≥ 0,8.

Transformer (src/transformer.py)
  distilbert-base-uncased avec une tête de classification multi-label (sigmoïde).
  Troncature à 256 tokens, lr 2e-5, batch de 16, 5 époques. Le meilleur checkpoint est
  choisi sur le F1 macro de validation. Décodage : chaque genre de probabilité ≥ 0,5.

Évaluation (src/evaluation.py)
  Les mêmes fonctions sont appliquées aux trois systèmes, sur le même test :
  - accuracy en exact match ;
  - précision, rappel et F1 macro ;
  - matrices de confusion ;
  - temps d'entraînement et d'inférence, mesurés dans les mêmes conditions ;
  - intervalles de confiance par bootstrap et test de McNemar ;
  - au moins 5 erreurs commentées, choisies par des critères fixés à l'avance et un
    tirage aléatoire (results/evaluation/erreurs.md).
  Le test ne sert qu'à l'évaluation finale. Tous les choix sont faits sur la validation.


7. Seeds et reproductibilité
----------------------------
- Seed 42 pour les modèles et l'évaluation : random, numpy et torch (y compris MPS et
  CUDA), set_seed de transformers, bootstrap et tirage des erreurs.
- Le découpage des données a été fait une fois avec random_state=67
  (notebooks/spliting.ipynb). Les fichiers sont versionnés et ne sont pas recalculés.
- Vérifications effectuées (results/evaluation/checks.json et dl_reproducibility.json) :
  - la baseline relancée donne des prédictions identiques ;
  - deux réentraînements du DL avec la seed 42 sur MPS donnent 100 % des mêmes
    prédictions de test ;
  - l'inférence du transformer refaite en CPU fp32 reproduit ses prédictions livrées.


8. Temps d'exécution indicatifs (Mac Apple Silicon)
---------------------------------------------------
  uv sync (premier lancement)           dépend du réseau (torch pèse plusieurs centaines de Mo)
  baseline                              quelques secondes
  entraînement DL                       environ 3 à 5 s (MPS)
  entraînement transformer              environ 7,5 min (MPS, bf16), bien plus long sur CPU
  src/run_evaluation.py                 environ 1 min
  src/analysis.py                       environ 20 s
  notebook complet (drapeaux par défaut) environ 20 s
