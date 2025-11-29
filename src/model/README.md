# Model Module

Module contenant tous les modeles de machine learning et deep learning pour le benchmark de volatility forecasting.

## Structure

```
model/
├── base.py                    # Interface de base (classe abstraite)
├── baseline/                  # Modeles de baseline
│   ├── persistence_baseline.py
│   ├── ar1_baseline.py
│   ├── random_baseline.py
│   └── main.py
├── ridge_classifier.py        # Ridge Regression
├── lasso_classifier.py        # Lasso Regression
├── elasticnet_classifier.py   # ElasticNet
├── logistic_classifier.py     # Logistic Regression
├── random_forest_model.py     # Random Forest
├── xgboost_model.py           # XGBoost
├── lightgbm_model.py          # LightGBM
├── catboost_model.py          # CatBoost
└── lstm_model.py              # LSTM (PyTorch)
```

## Interface BaseModel

Tous les modeles heritent de `BaseModel` et implementent :

```python
class BaseModel(ABC):
    def fit(self, X, y, **kwargs) -> BaseModel:
        """Entraine le modele."""
        pass

    def predict(self, X) -> np.ndarray:
        """Fait des predictions."""
        pass

    def save(self, path: Path) -> None:
        """Sauvegarde le modele."""
        pass

    @classmethod
    def load(cls, path: Path) -> BaseModel:
        """Charge un modele sauvegarde."""
        pass
```

## Categories de Modeles

### Baselines
- **Persistence** : Predit la derniere valeur observee
- **AR(1)** : Modele autoregressif d'ordre 1
- **Random** : Predictions aleatoires (lower bound)

### Modeles Econometriques (Lineaires)
- **Ridge** : Regression avec regularisation L2
- **Lasso** : Regression avec regularisation L1
- **ElasticNet** : Combinaison L1 + L2
- **Logistic** : Classification logistique

### Machine Learning (Tree-based)
- **Random Forest** : Ensemble de decision trees
- **XGBoost** : Gradient boosting optimise
- **LightGBM** : Gradient boosting rapide
- **CatBoost** : Gradient boosting avec gestion des categoriques

### Deep Learning
- **LSTM** : Long Short-Term Memory (PyTorch)
  - Architecture legere adaptee aux series temporelles
  - Dropout pour regularisation

## Utilisation

```python
from src.model.xgboost_model import XGBoostModel

# Initialisation
model = XGBoostModel(
    name="xgb_volatility",
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
)

# Entrainement
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Sauvegarde
model.save(Path("models/xgb_model.joblib"))

# Chargement
loaded_model = XGBoostModel.load(Path("models/xgb_model.joblib"))
```

## Baselines Pipeline

```bash
python -m src.model.baseline.main
```

Evalue tous les baselines sur les donnees de test.

## Notes d'Implementation

### Tree-based (XGBoost, LightGBM, CatBoost)
- Ne necessitent PAS de normalisation des features
- Utilisent `dataset_features.parquet` directement

### Lineaires (Ridge, Lasso)
- Necessitent normalisation z-score
- Utilisent `dataset_features_linear.parquet`

### LSTM
- Necessite normalisation min-max [-1, 1]
- Utilise `dataset_features_lstm.parquet`
- Sequences temporelles (lookback window)

## Serialisation

Les modeles sont sauvegardes avec `joblib` pour :
- Persistance des poids
- Reproductibilite
- Deploiement
