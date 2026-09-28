| Méthode | accuracy | precision_macro | recall_macro | f1_macro | f1_micro | n_examples | n_empty_predictions | n_multi_predictions | labels_per_example |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 0.452 | 0.446 | 0.638 | 0.486 | 0.505 | 447 | 0 | 57 | 1.174 |
| Deep Learning | 0.700 | 0.789 | 0.678 | 0.716 | 0.717 | 447 | 0 | 2 | 1.004 |
| Transformer | 0.655 | 0.770 | 0.592 | 0.658 | 0.722 | 447 | 65 | 3 | 0.861 |
| Baseline, décodage harmonisé (seuil top-1 seul) | 0.494 | 0.482 | 0.602 | 0.487 | 0.504 | 447 | 0 | 0 | 1.000 |
| Deep Learning, décodage harmonisé (seuil 0.6) | 0.678 | 0.776 | 0.719 | 0.737 | 0.720 | 447 | 0 | 25 | 1.058 |
| Transformer, décodage harmonisé (seuil 0.2) | 0.609 | 0.720 | 0.772 | 0.734 | 0.717 | 447 | 0 | 102 | 1.237 |
