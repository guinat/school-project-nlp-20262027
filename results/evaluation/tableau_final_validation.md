| Méthode | Accuracy | Précision | Rappel | F1 macro | Train | Inférence |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | 0.452 | 0.446 | 0.638 | 0.486 | 0.000 | 1.149 |
| Deep Learning | 0.700 | 0.789 | 0.678 | 0.716 | 2.949 | 0.121 |
| Transformer | 0.655 | 0.770 | 0.592 | 0.658 | 451.492 | 10.269 |
| Baseline, décodage harmonisé (seuil top-1 seul) | 0.494 | 0.482 | 0.602 | 0.487 | 0.000 | 1.149 |
| Deep Learning, décodage harmonisé (seuil 0.6) | 0.678 | 0.776 | 0.719 | 0.737 | 2.949 | 0.121 |
| Transformer, décodage harmonisé (seuil 0.2) | 0.609 | 0.720 | 0.772 | 0.734 | 451.492 | 10.269 |
