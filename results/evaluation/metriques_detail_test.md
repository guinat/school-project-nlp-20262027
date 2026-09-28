| Méthode | accuracy | precision_macro | recall_macro | f1_macro | f1_micro | n_examples | n_empty_predictions | n_multi_predictions | labels_per_example |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 0.415 | 0.426 | 0.613 | 0.464 | 0.475 | 508 | 0 | 78 | 1.203 |
| Deep Learning | 0.701 | 0.764 | 0.707 | 0.730 | 0.711 | 508 | 0 | 5 | 1.010 |
| Transformer | 0.650 | 0.754 | 0.584 | 0.654 | 0.718 | 508 | 83 | 3 | 0.843 |
| Baseline, décodage harmonisé (seuil top-1 seul) | 0.455 | 0.452 | 0.559 | 0.449 | 0.458 | 508 | 0 | 0 | 1.000 |
| Deep Learning, décodage harmonisé (seuil 0.6) | 0.671 | 0.725 | 0.719 | 0.719 | 0.711 | 508 | 0 | 29 | 1.059 |
| Transformer, décodage harmonisé (seuil 0.2) | 0.583 | 0.697 | 0.783 | 0.732 | 0.721 | 508 | 0 | 134 | 1.270 |
