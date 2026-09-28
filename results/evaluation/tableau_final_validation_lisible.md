| Méthode | Accuracy | Précision | Rappel | F1 macro | Train | Inférence |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | 0.452 | 0.446 | 0.638 | 0.486 | 0 s | 1.15 ms/ex. |
| Deep Learning | 0.700 | 0.789 | 0.678 | 0.716 | 2.9 s | 0.12 ms/ex. |
| Transformer | 0.655 | 0.770 | 0.592 | 0.658 | 451.5 s | 10.27 ms/ex. |
| Baseline, décodage harmonisé (seuil top-1 seul) | 0.494 | 0.482 | 0.602 | 0.487 | 0 s | 1.15 ms/ex. |
| Deep Learning, décodage harmonisé (seuil 0.6) | 0.678 | 0.776 | 0.719 | 0.737 | 2.9 s | 0.12 ms/ex. |
| Transformer, décodage harmonisé (seuil 0.2) | 0.609 | 0.720 | 0.772 | 0.734 | 451.5 s | 10.27 ms/ex. |
