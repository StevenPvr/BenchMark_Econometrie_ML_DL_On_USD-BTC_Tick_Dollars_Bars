# Utils Module

Fonctions utilitaires partagees pour le traitement de donnees, validation, I/O et calculs financiers.

## Fichiers

| Fichier | Description |
|---------|-------------|
| `validation.py` | Validation de DataFrames, fichiers et parametres |
| `temporal.py` | Validation temporelle et split train/test |
| `io.py` | Operations I/O (CSV, Parquet, JSON) |
| `transforms.py` | Transformations de donnees |
| `datetime_utils.py` | Parsing et manipulation de dates |
| `metrics.py` | Metriques statistiques et financieres |
| `logging_utils.py` | Logging et plotting |
| `financial.py` | Calculs financiers (poids de liquidite) |
| `statsmodels_utils.py` | Utilitaires statsmodels |
| `user_input.py` | Gestion des inputs utilisateur |

## Categories de Fonctions

### Validation (`validation.py`)
```python
validate_dataframe_not_empty(df)      # Verifie DataFrame non vide
validate_file_exists(path)             # Verifie existence fichier
validate_required_columns(df, cols)    # Verifie colonnes requises
validate_series(series, name)          # Valide une Series
validate_train_ratio(ratio)            # Valide ratio train/test
has_both_splits(df)                    # Verifie train/test presents
```

### Temporal (`temporal.py`)
```python
compute_timeseries_split_indices(n, ratio)  # Indices de split
validate_temporal_order_series(series)       # Ordre chronologique
validate_temporal_split(df_train, df_test)   # Split sans overlap
log_split_dates(df_train, df_test)           # Log des plages de dates
```

### I/O (`io.py`)
```python
ensure_output_dir(path)                # Cree repertoire si necessaire
load_dataframe(path)                   # Charge DataFrame (auto-detect format)
load_parquet_file(path)                # Charge Parquet
load_csv_file(path)                    # Charge CSV
load_json_data(path)                   # Charge JSON
save_parquet_and_csv(df, path)         # Sauvegarde dual format
save_json_pretty(data, path)           # Sauvegarde JSON formatte
```

### Transforms (`transforms.py`)
```python
extract_features_and_target(df, target)  # Separe X et y
filter_by_split(df, split)               # Filtre train/test
remove_metadata_columns(df)              # Supprime colonnes meta
stable_ticker_id(df)                     # ID ticker stable
```

### DateTime (`datetime_utils.py`)
```python
parse_date_value(value)                  # Parse date flexible
normalize_timestamp_to_datetime(ts)      # Normalise timestamp
filter_by_date_range(df, start, end)     # Filtre par dates
extract_date_range(df)                   # Extrait min/max dates
format_dates_to_string(dates)            # Formate dates en string
```

### Metrics (`metrics.py`)
```python
compute_log_returns(prices)              # Calcule log-returns
compute_residuals(actual, predicted)     # Calcule residus
chi2_sf(x, df)                           # Survival function chi2
```

### Logging (`logging_utils.py`)
```python
log_series_summary(series, name)         # Resume d'une Series
log_split_summary(df_train, df_test)     # Resume du split
save_plot(fig, path)                     # Sauvegarde figure
```

### Financial (`financial.py`)
```python
compute_rolling_volume_scaling(df, window)  # Poids de volume rolling
```

## Utilisation

```python
from src.utils import (
    validate_dataframe_not_empty,
    ensure_output_dir,
    load_dataframe,
    compute_log_returns,
    get_logger,
)

# Logger
logger = get_logger(__name__)

# Validation
validate_dataframe_not_empty(df)

# I/O
ensure_output_dir(Path("outputs/results"))
df = load_dataframe(Path("data/features.parquet"))

# Calculs
log_returns = compute_log_returns(df["close"])
```

## Bonnes Pratiques

1. **Validation** : Toujours valider les inputs au debut des fonctions
2. **Logging** : Utiliser `get_logger(__name__)` pour tracer l'execution
3. **Paths** : Utiliser `Path` de `pathlib` plutot que des strings
4. **Types** : Les fonctions sont typees pour meilleure documentation
