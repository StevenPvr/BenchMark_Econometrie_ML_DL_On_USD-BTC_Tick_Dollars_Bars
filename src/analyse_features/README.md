# Analyse Features Module

Module d'analyse comprehensive des features pour le projet de volatility forecasting.

## Fichiers

| Fichier | Description |
|---------|-------------|
| `main.py` | Orchestrateur principal du pipeline d'analyse |
| `correlation.py` | Analyse de correlation (Spearman, dCor, MI) |
| `stationarity.py` | Tests de stationnarite (ADF, KPSS) |
| `multicollinearity.py` | Analyse multicollinearite (VIF, Condition Number) |
| `target_analysis.py` | Relations feature-target |
| `clustering.py` | Clustering (Hierarchical, t-SNE, UMAP) |
| `temporal.py` | Analyse temporelle (ACF/PACF, correlations rolling) |
| `config.py` | Configuration et chemins |
| `utils/` | Utilitaires (plotting, parallel, json, numba) |

## Analyses Disponibles

### 1. Correlation Analysis
- **Spearman** : Correlation de rang (robuste aux outliers)
- **Distance Correlation (dCor)** : Capture les dependances non-lineaires
- **Mutual Information (MI)** : Information mutuelle normalisee

### 2. Stationarity Analysis
- **ADF (Augmented Dickey-Fuller)** : Test de racine unitaire
- **KPSS** : Test de stationnarite autour d'une tendance
- Classification : Stationnaire, Non-stationnaire, Ambigu

### 3. Multicollinearity Analysis
- **VIF (Variance Inflation Factor)** : Detection de collinearite
- **Condition Number** : Stabilite numerique de la matrice de correlation
- Seuil VIF > 10 = collinearite problematique

### 4. Target Analysis
- Correlations feature-target (log_return)
- Classement des features par predictabilite
- Identification des features les plus informatives

### 5. Clustering Analysis
- **Hierarchical** : Dendrogramme des features similaires
- **t-SNE** : Visualisation 2D des relations
- **UMAP** : Embedding pour clusters locaux

### 6. Temporal Analysis
- **ACF/PACF** : Autocorrelation des features
- **Rolling Correlations** : Stabilite temporelle des relations
- Detection de regimes

## Utilisation

```bash
# Toutes les analyses
python -m src.analyse_features.main

# Analyse specifique
python -m src.analyse_features.main --analysis correlation

# Skip analyses lentes
python -m src.analyse_features.main --skip-dcor --skip-umap
```

## Configuration

Dans `config.py` :
- `TARGET_COLUMN` : Colonne cible (defaut: "log_return")
- `DATASET_SAMPLE_FRACTION` : Fraction d'echantillonnage
- `ANALYSE_FEATURES_DIR` : Repertoire de sortie

## Sorties

Fichiers JSON generes :
- `correlation_results.json` : Matrices de correlation
- `stationarity_results.json` : Resultats ADF/KPSS
- `multicollinearity_results.json` : VIF et condition number
- `target_results.json` : Correlations avec la target
- `clustering_results.json` : Labels et embeddings
- `temporal_results.json` : ACF/PACF
- `summary.json` : Resume global

Plots generes :
- Heatmaps de correlation
- Scores VIF
- Dendrogrammes
- Embeddings t-SNE/UMAP
- Correlations avec la target
