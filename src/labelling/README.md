# Labelling Module

Module de labellisation selon la methodologie De Prado (AFML Chapter 3).

## Structure

```
labelling/
├── label_primaire/    # Modele primaire (prediction direction)
│   ├── main.py        # Point d'entree
│   ├── opti.py        # Optimisation hyperparametres
│   ├── train.py       # Entrainement
│   ├── eval.py        # Evaluation
│   └── utils.py       # Utilitaires
├── label_meta/        # Meta-labelling (bet sizing)
│   ├── opti.py        # Optimisation
│   └── utils.py       # Utilitaires
└── __init__.py
```

## Concepts De Prado

### Label Primaire (Primary Model)

Le modele primaire predit la DIRECTION du trade :

- **+1 (Long)** : Prix va augmenter
- **-1 (Short)** : Prix va baisser
- **0 (Neutral)** : Pas de mouvement significatif

### Meta-Labelling (Chapter 3.6)

Le meta-labelling determine le BET SIZING :

- Utilise les predictions du modele primaire
- Estime la probabilite de succes
- Ajuste la taille de position en consequence

## Triple Barrier Method

Methode de labellisation basee sur 3 barrieres :

1. **Take Profit (PT)** : Seuil de profit
2. **Stop Loss (SL)** : Seuil de perte
3. **Time Barrier (T1)** : Horizon temporel max

Le label est determine par la premiere barriere touchee.

## Pipeline

```bash
# 1. Optimisation des hyperparametres
python -m src.labelling.label_primaire.opti

# 2. Entrainement
python -m src.labelling.label_primaire.train

# 3. Evaluation
python -m src.labelling.label_primaire.eval
```

## Fonctions Principales

### get_daily_volatility

Calcul de la volatilite journaliere pour dimensionner les barrieres.

### apply_pt_sl_on_t1

Application de la methode triple barrier :

- PT/SL en multiples de volatilite
- Horizon temporel adaptatif

### get_events_primary

Generation des events pour le modele primaire :

- Labels (-1, 0, +1)
- Temps de premiere barriere touchee

### optimize_model

Optimisation Optuna des hyperparametres :

- Parametres du modele ML
- Parametres triple barrier (PT, SL, T1)

### WalkForwardCV

Cross-validation walk-forward pour respecter la temporalite.

## Modeles Supportes

Le `MODEL_REGISTRY` contient :

- Ridge, Lasso (modeles lineaires)
- Random Forest
- XGBoost, LightGBM, CatBoost
- LSTM

## Configuration

Dans `utils.py` :

- `TRIPLE_BARRIER_SEARCH_SPACE` : Espace de recherche Optuna
- `OptimizationConfig` : Configuration optimisation
- `TrainingConfig` : Configuration entrainement

## Reference

Lopez de Prado, M. (2018). Advances in Financial Machine Learning. Chapter 3: Labeling.
