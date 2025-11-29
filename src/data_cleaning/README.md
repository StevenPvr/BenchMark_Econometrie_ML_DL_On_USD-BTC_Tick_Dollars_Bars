# Data Cleaning Module

Pipeline de nettoyage des donnees tick pour les trades crypto (sortie ccxt).

## Fichiers

| Fichier | Description |
|---------|-------------|
| `cleaning.py` | Logique de nettoyage avec detection robuste d'outliers |
| `main.py` | Point d'entree CLI |
| `__init__.py` | Exports du module |

## Methodes de Detection d'Outliers

Le module implemente 5 methodes robustes adaptees aux marches financiers :

### 1. MAD (Median Absolute Deviation)
- Robuste aux distributions a queues epaisses (fat-tailed)
- Facteur de scaling 1.4826 pour compatibilite avec l'ecart-type
- Reference : Huber, P.J. (1981). Robust Statistics

### 2. Rolling Z-score
- S'adapte aux regimes de volatilite locaux
- Utilise median et MAD roulants pour la robustesse
- Tolere des mouvements plus grands en haute volatilite

### 3. Detection Flash Crash/Spike
- Identifie les anomalies de prix transitoires
- Criteres : mouvement > 15% avec reversion rapide
- Filtre les erreurs de donnees et gaps de liquidite

### 4. Anomalies de Volume
- Filtre les "dust trades" (volume minimum)
- Detection MAD pour volumes extremes (manipulation)

### 5. Filtrage Dollar Value
- Combine prix * volume pour detecter les anomalies
- Identifie : fat-finger errors, wash trading, glitches

## Ordre d'Application

1. Suppression dust trades (volume minimum)
2. Outliers prix MAD (anomalies globales)
3. Rolling Z-score (ajuste volatilite locale)
4. Detection flash spikes (erreurs transitoires)
5. Outliers volume (MAD)
6. Outliers dollar value (anomalies combinees)

## Utilisation

```bash
python -m src.data_cleaning.main
```

## Configuration

Seuils definis dans `src/constants.py` :
- `OUTLIER_MAD_THRESHOLD` : Seuil MAD pour prix
- `OUTLIER_ROLLING_WINDOW` : Fenetre rolling Z-score
- `OUTLIER_MAX_TICK_RETURN` : Retour max entre ticks (15%)
- `OUTLIER_MIN_VOLUME` : Volume minimum
- `OUTLIER_VOLUME_MAD_THRESHOLD` : Seuil MAD volume

## Sortie

- Entree : `DATASET_RAW_PARQUET` (donnees brutes)
- Sortie : `DATASET_CLEAN_PARQUET` (donnees nettoyees)

Un rapport `OutlierReport` est genere avec les statistiques detaillees.
