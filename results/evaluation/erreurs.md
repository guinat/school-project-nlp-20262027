# Analyse d'erreurs (test, prédictions telles que livrées)

Sélection automatique : pour chaque critère (ordre fixe), tirage aléatoire avec `numpy.random.default_rng(42)` parmi les exemples qui le vérifient, sans remise entre critères. Aucun exemple choisi à la main.

| Critère | Candidats | Retenus |
| --- | --- | --- |
| A. les trois systèmes se trompent | 88 | 2 |
| B. seule la baseline se trompe | 128 | 1 |
| C. le transformer ne prédit aucun genre | 83 | 1 |
| D. confusion thriller / crime / horror (DL ou transformer) | 61 | 1 |
| E. confusion fantasy / horror (DL ou transformer) | 18 | 1 |
| F. exemple à plusieurs genres | 6 | 1 |
| G. genre douteux : DL et transformer d'accord contre l'étiquette, tous deux à p >= 0.8 | 7 | 1 |
| H. petite classe manquée par le DL | 13 | 1 |

## Erreur 1 — test #38 (A. les trois systèmes se trompent)

> After four teenagers mysteriously die simultaneously in Tokyo, Kazuyuki Asakawa, a reporter and uncle to one of the deceased, decides to launch his own personal investigation. His search leads him to …

- Longueur : 274 mots
- **Vrai genre** : horror
- Baseline : crime — mots-clés : crime 2 (investigation, witnesses)
- Deep Learning : thriller — probabilités : thriller 0.37, crime 0.30, horror 0.22
- Transformer : (aucun) — probabilités : horror 0.49, thriller 0.47, crime 0.08
- Transformer, décodage harmonisé : horror, thriller
- **Hypothèse** : Résumé de *Ring* (K. Suzuki) écrit comme une enquête (reporter, « investigation », « witnesses ») : l'horreur n'est portée que par la cassette maudite, sans mot-clé explicite (ghost, haunted). La baseline tombe dans le piège lexical (crime). Le transformer place bien horror en tête (0,49) mais juste sous le seuil de 0,5 : il ne prédit rien.

## Erreur 2 — test #379 (A. les trois systèmes se trompent)

> The novel chronicles the stay of an unnamed weather official on a remote island in the south Atlantic close to the Antarctic Circle.

- Longueur : 23 mots
- **Vrai genre** : horror
- Baseline : travel — mots-clés : travel 1 (island)
- Deep Learning : thriller — probabilités : thriller 0.64, romance 0.25, travel 0.23
- Transformer : (aucun) — probabilités : science 0.18, thriller 0.16, horror 0.07
- Transformer, décodage harmonisé : science
- **Hypothèse** : Résumé de 23 mots sans aucun indice de genre (*La Peau froide*, roman d'horreur) : aucune méthode fondée sur le texte seul ne peut trouver horror. La baseline s'accroche à « island » (travel). Limite de la donnée, pas du modèle.

## Erreur 3 — test #236 (B. seule la baseline se trompe)

> March 1997. A woman has her throat cut behind a bar in Carter Crossing, Mississippi. Just down the road is a big army base. Is the murderer a local guy - or is he a soldier?  Jack Reacher, still a maj…

- Longueur : 145 mots
- **Vrai genre** : thriller
- Baseline : crime — mots-clés : crime 4.5 (murderer, investigation, police); history 1 (soldier); thriller 1 (killer)
- Deep Learning : thriller — probabilités : thriller 0.89, crime 0.20, horror 0.18
- Transformer : thriller — probabilités : thriller 0.94, romance 0.04, fantasy 0.03
- Transformer, décodage harmonisé : thriller
- **Hypothèse** : Piège lexical : le vocabulaire de l'enquête (murderer, investigation, police) donne 4,5 points à crime contre 1 à thriller (« killer », mot faible). Ce qui fait de ce livre un thriller (série Jack Reacher, action) n'est pas dans le lexique. Le DL et le transformer ont appris cette frontière sur les données et ont raison.

## Erreur 4 — test #219 (C. le transformer ne prédit aucun genre)

> Lane Roanoke is fifteen when she comes to live with her grandparents and fireball cousin at the Roanoke family's rural estate following the suicide of her mother. Over one long, hot summer, Lane exper…

- Longueur : 89 mots
- **Vrai genre** : thriller
- Baseline : romance, horror — mots-clés : horror 1 (darkness); romance 1 (heart)
- Deep Learning : romance — probabilités : romance 0.48, horror 0.48, fantasy 0.26
- Transformer : (aucun) — probabilités : horror 0.35, thriller 0.17, fantasy 0.14
- Transformer, décodage harmonisé : horror
- **Hypothèse** : Le suspense est suggéré (« terrible legacy », « either the girls run, or they die », « darkness ») mais jamais nommé. Les indices vont autant vers horror que vers romance : le DL hésite (0,48 / 0,48) et le transformer ne dépasse 0,5 pour aucun genre. Frontière floue entre thriller psychologique et horreur.

## Erreur 5 — test #436 (D. confusion thriller / crime / horror (DL ou transformer))

> The story takes place in 1960s suburban United States, and is told in flashback form by the narrator, David. After giving the reader a quick tour of his neighborhood and childhood friends, David intro…

- Longueur : 430 mots
- **Vrai genre** : thriller
- Baseline : travel — mots-clés : travel 3.5 (travel, tour); crime 2 (crime, police)
- Deep Learning : horror — probabilités : horror 0.39, history 0.22, science 0.19
- Transformer : horror — probabilités : horror 0.68, thriller 0.15, crime 0.04
- Transformer, décodage harmonisé : horror
- **Hypothèse** : *The Girl Next Door* (J. Ketchum) est souvent classé en horreur : la prédiction horror du DL et du transformer est défendable et l'étiquette thriller discutable. La baseline est piégée par « a quick tour of his neighborhood » et « travel » : le lexique ignore le contexte.

## Erreur 6 — test #116 (E. confusion fantasy / horror (DL ou transformer))

> Firmin, is a rat and the runt son of an alcoholic mother living in 1960s Boston, has been forced to survive on the pages of novels from the bookstore in which he dwells. At first everything holds the …

- Longueur : 57 mots
- **Vrai genre** : fantasy
- Baseline : thriller — mots-clés : aucun mot-clé, genre par défaut (thriller)
- Deep Learning : horror — probabilités : horror 0.40, science 0.33, fantasy 0.27
- Transformer : horror — probabilités : horror 0.64, thriller 0.09, fantasy 0.08
- Transformer, décodage harmonisé : horror
- **Hypothèse** : *Firmin* (S. Savage), fable littéraire sur un rat qui apprend à lire. Aucun mot du lexique : la baseline renvoie le genre par défaut. Le fantastique vient d'un animal qui lit, ce qui demande de comprendre le sens ; « rat », « alcoholic », « survive » tirent les deux réseaux vers horror. L'étiquette fantasy est elle-même discutable.

## Erreur 7 — test #481 (F. exemple à plusieurs genres)

> When fifteen-year-old Clary Fray heads out to the Pandemonium Club in New York City, she hardly expects to witness a murder― much less a murder committed by three teenagers covered with strange tattoo…

- Longueur : 171 mots
- **Vrai genre** : romance, fantasy
- Baseline : crime — mots-clés : crime 7 (murder, murderers, police, witness); fantasy 3 (demon, demons, warriors); horror 1 (blood)
- Deep Learning : fantasy — probabilités : fantasy 0.55, horror 0.43, romance 0.33
- Transformer : fantasy — probabilités : fantasy 0.75, horror 0.18, thriller 0.03
- Transformer, décodage harmonisé : fantasy
- **Hypothèse** : *City of Bones* (C. Clare), étiqueté romance + fantasy. Les deux réseaux trouvent fantasy mais pas romance, seulement suggérée par un personnage (« looks a little like an angel and acts a lot like a jerk »). La baseline est piégée par le vocabulaire du meurtre (crime 7 points). Les exemples multi-genres sont rares à l'entraînement : les modèles prédisent surtout un seul genre.

## Erreur 8 — test #136 (G. genre douteux : DL et transformer d'accord contre l'étiquette, tous deux à p >= 0.8)

> He was supposed to be a myth. But from the moment I crossed the River Styx and fell under his dark spell... he was, quite simply, mine.  Society darling Persephone Dimitriou plans to flee the ultra-mo…

- Longueur : 198 mots
- **Vrai genre** : romance
- Baseline : fantasy — mots-clés : fantasy 2.5 (spell); history 1 (war); thriller 1 (dangerous)
- Deep Learning : fantasy — probabilités : fantasy 0.85, romance 0.27, thriller 0.20
- Transformer : fantasy — probabilités : fantasy 0.94, romance 0.06, horror 0.04
- Transformer, décodage harmonisé : fantasy
- **Hypothèse** : *Neon Gods* (K. Robert), réécriture de Hadès et Perséphone : une romance dans un décor mythologique. Les trois systèmes prédisent fantasy, les réseaux avec confiance (0,85 et 0,94). L'étiquette unique romance est incomplète plutôt que fausse : romance + fantasy aurait été plus juste. Erreur de la donnée plus que du modèle.

## Erreur 9 — test #12 (H. petite classe manquée par le DL)

> A fresh, urban twist on the classic tale of star-crossed lovers.  When Brittany Ellis walks into chemistry class on the first day of senior year, she has no clue that her carefully created 'perfect' l…

- Longueur : 171 mots
- **Vrai genre** : romance
- Baseline : romance — mots-clés : romance 4.5 (passionate, boyfriend, chemistry); crime 1 (clue); thriller 1 (secret)
- Deep Learning : thriller — probabilités : thriller 0.59, romance 0.38, sports 0.26
- Transformer : (aucun) — probabilités : romance 0.38, thriller 0.29, sports 0.19
- Transformer, décodage harmonisé : romance, thriller
- **Hypothèse** : *Perfect Chemistry* (S. Elkeles) : la baseline a raison (passionate, boyfriend, chemistry). Le DL préfère thriller (gang, secret) ; le transformer place romance en tête (0,38) mais sous le seuil de 0,5. romance est une petite classe (88 résumés d'entraînement sur 3 580) : les réseaux y sont peu confiants.
