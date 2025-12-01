# Clear Features Module

Module de transformation des features : reduction PCA, log transform, et normalisation.

## Fichiers

| Fichier | Description |
|---------|-------------|
| `main.py` | Pipeline principal de transformation |
| `pca_reducer.py` | Reduction PCA par groupes de features |
| `log_transformer.py` | Transformation logarithmique |
| `scaler_applier.py` | Application des scalers (z-score, minmax) |
| `nonlinear_correlation.py` | Detection de correlations non-lineaires |
| `config.py` | Configuration et parametres |
| `__init__.py` | Exports du module |

## Pipeline de Transformation

```
features/main.py -> clear_features/main.py -> ready for training
```

### Etapes

1. **Standardisation (Z-score)** : Appliquee avant PCA
2. **Normalisation (Min-Max)** : Pour LSTM
3. **Reduction PCA par groupes** : Fit sur TRAIN, transform sur tous

## Classes Principales

### GroupPCAReducer

Reduction PCA intelligente par groupes de features similaires :

- Detecte automatiquement les groupes correles
- Fit uniquement sur les donnees TRAIN (anti-leakage)
- Preserve la variance expliquee cible

### LogTransformer

Transformation logarithmique pour features non-stationnaires :

- Detecte les features qui beneficient du log
- Gere les valeurs negatives (shift)

### ScalerApplier

Application des scalers pre-fits :

- Z-score pour modeles lineaires (Ridge, Lasso)
- Min-Max [-1, 1] pour LSTM

## Anti-Leakage

**CRITIQUE** : Toutes les transformations sont :

1. **Fit sur TRAIN uniquement**
2. **Transform sur TRAIN et TEST**

Cela evite tout leakage d'information du test vers le train.

## Utilisation

```bash
python -m src.clear_features.main
```

## Configuration

Dans `config.py` :

- `PCA_CONFIG` : Parametres PCA (variance cible, groupes)
- `TARGET_COLUMN` : Colonne a ne pas transformer
- `META_COLUMNS` : Colonnes meta a preserver

## Entrees / Sorties

**Entrees** :

- `dataset_features.parquet` : Features brutes (tree-based)
- `dataset_features_linear.parquet` : Pour modeles lineaires
- `dataset_features_lstm.parquet` : Pour LSTM

**Sorties** (ecrase les fichiers) :

- Memes fichiers avec features transformees
- Artefacts PCA dans `PCA_ARTIFACTS_DIR`

## Flux de Donnees

```
Raw Features
     |
     v
[Z-score Standardization] (linear only)
     |
     v
[Min-Max Normalization] (LSTM only)
     |
     v
[Group PCA Reduction]
     |
     v
Transformed Features
```
