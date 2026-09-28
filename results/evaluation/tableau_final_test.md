| Méthode | Accuracy | Précision | Rappel | F1 macro | Train | Inférence |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | 0.415 | 0.426 | 0.613 | 0.464 | 0.000 | 1.149 |
| Deep Learning | 0.701 | 0.764 | 0.707 | 0.730 | 2.949 | 0.121 |
| Transformer | 0.650 | 0.754 | 0.584 | 0.654 | 451.492 | 10.269 |
| Baseline, décodage harmonisé (seuil top-1 seul) | 0.455 | 0.452 | 0.559 | 0.449 | 0.000 | 1.149 |
| Deep Learning, décodage harmonisé (seuil 0.6) | 0.671 | 0.725 | 0.719 | 0.719 | 2.949 | 0.121 |
| Transformer, décodage harmonisé (seuil 0.2) | 0.583 | 0.697 | 0.783 | 0.732 | 451.492 | 10.269 |
