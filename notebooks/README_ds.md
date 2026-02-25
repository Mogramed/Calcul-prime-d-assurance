# DS notebooks order

1. `07_ds_cadrage_qualite_cv.ipynb`
2. `08_ds_eda_segmentation_preprocessing.ipynb`
3. `09_ds_model_diagnostics_storytelling.ipynb`

## Objectif

Suite de notebooks "démarche data science" pour:
- cadrage métier assurance,
- compréhension et qualité des données,
- EDA / segmentation / preprocessing,
- diagnostics modèle / robustesse / storytelling de soutenance.

## Pré-requis

- Notebook 07 et 08: utilisables directement avec `data/train.csv` et `data/test.csv`
- Notebook 09: idéalement après génération des artefacts V2 (`artifacts/v2/*`)

## Modes d'exécution

Les notebooks exposent:
- `QUICK_ANALYSIS = True`: exécution rapide sur sous-échantillons
- `FULL_ANALYSIS = False`: passer à `True` pour analyses plus lourdes

## Temps estimés (ordre de grandeur)

- Notebook 07: 5-15 min
- Notebook 08: 10-30 min (selon clustering / plots)
- Notebook 09: 10-40 min (selon taille artefacts OOF)

## Sorties

Les tables analytiques sont exportées sous `artifacts/ds/`.
