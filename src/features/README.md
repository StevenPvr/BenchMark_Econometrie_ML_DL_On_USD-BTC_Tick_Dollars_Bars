# Features Module

Module de calcul des features de trading a partir des dollar bars.

## Fichiers

| Fichier | Description |
|---------|-------------|
| `main.py` | Pipeline principal de feature engineering |
| `__init__.py` | Exports de toutes les fonctions |
| `momentum.py` | Returns cumules et extremes recents |
| `realized_volatility.py` | Volatilite historique et Sharpe local |
| `trend.py` | Moyennes mobiles, z-score, cross MA |
| `range_volatility.py` | Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang |
| `temporal_acceleration.py` | Acceleration de formation des bars |
| `temporal_calendar.py` | Encodage cyclique du temps, regimes |
| `entropy.py` | Shannon, Approximate, Sample entropy |
| `vpin.py` | Volume-Synchronized Probability of Informed Trading |
| `kyle_lambda.py` | Coefficient d'impact de prix |
| `volume_imbalance.py` | Desequilibre de volume (toxicite) |
| `trade_classification.py` | Classification buy/sell des trades |
| `microstructure_volatility.py` | Volatilite intrabar |
| `fractional_diff.py` | Differentiation fractionnaire (De Prado FFD) |
| `lag_generator.py` | Generation intelligente des lags |
| `zscore_normalizer.py` | Normalisation z-score rolling |
| `scalers.py` | StandardScaler et MinMaxScaler custom |
| `technical_indicators.py` | Indicateurs d'analyse technique (ta library) |

## Categories de Features

### 1. Momentum
- Returns cumules sur fenetres multiples
- Extremes recents (max/min)

### 2. Volatilite Realisee
- Ecart-type rolling des returns
- Sharpe ratio local

### 3. Tendance
- Moyennes mobiles (SMA, EMA)
- Z-score du prix vs MA
- Cross MA (signaux)
- Return streak (sequences)

### 4. Volatilite Range-Based
- Parkinson : volatilite basee sur High-Low
- Garman-Klass : inclut Open-Close
- Rogers-Satchell : drift-independent
- Yang-Zhang : combine overnight et intraday

### 5. Acceleration Temporelle
- Vitesse de formation des bars
- Jerk (derivee seconde)

### 6. Microstructure / Order Flow
- Volume Imbalance : (buy - sell) / total
- VPIN : Probabilite de trading informe
- Kyle's Lambda : Impact de prix

### 7. Entropie
- Shannon : complexite de la distribution
- Approximate/Sample : regularite de la serie

### 8. Calendrier / Regime
- Encodage cyclique (heure, jour, mois)
- Detection regime de volatilite
- Drawdown features

### 9. Differentiation Fractionnaire
- FFD (Fixed-width window Fractionally Differenced)
- Preserve la memoire longue tout en rendant stationnaire

## Pipeline

```bash
python -m src.features.main
```

### Etapes du Pipeline

1. Chargement des dollar bars
2. Calcul de toutes les features
3. Application de la structure de lags intelligente
4. Nettoyage des NaN (debut de serie)
5. Shift de la target (predict next return)
6. Split train/test (80/20)
7. Fit des scalers sur TRAIN uniquement
8. Sauvegarde des datasets

## Configuration Anti-Leakage

- **Scalers** : Fit sur TRAIN, transform sur TRAIN et TEST
- **Target shift** : log_return shifted de -1 (predict next bar)
- **Lags** : Pas d'utilisation de donnees futures

## Sortie

3 datasets generes :
- `dataset_features.parquet` : Raw (tree-based ML)
- `dataset_features_linear.parquet` : Pour modeles lineaires
- `dataset_features_lstm.parquet` : Pour LSTM

Scalers sauvegardes dans `SCALERS_DIR` :
- `zscore_scaler.joblib`
- `minmax_scaler.joblib`
