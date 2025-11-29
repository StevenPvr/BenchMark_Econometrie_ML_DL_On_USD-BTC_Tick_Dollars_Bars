# Data Preparation Module

Implementation des Dollar Bars selon la methodologie de De Prado (AFML Chapter 2).

## Fichiers

| Fichier | Description |
|---------|-------------|
| `preparation.py` | Algorithme Dollar Bars optimise avec Numba |
| `main.py` | Point d'entree CLI |
| `__init__.py` | Exports du module |

## Dollar Bars (De Prado)

Les Dollar Bars echantillonnent les donnees chaque fois qu'une valeur monetaire predefinite T est echangee, plutot qu'a intervalles de temps fixes.

### Avantages
- Plus de bars pendant les periodes de haute activite (marches volatils)
- Moins de bars pendant les periodes calmes
- Proprietes statistiques ameliorees (plus proche de IID Gaussien)
- Synchronisation avec le flux d'information du marche

### Formulation Mathematique

Pour chaque tick t avec prix p_t et volume v_t :
- Dollar value : `dv_t = p_t * v_t`
- La bar k se ferme au tick t quand : `sum(dv_i for i in [t_start, t]) >= T_k`

### Calibration (De Prado Expected Value)

```
T = Total Dollar Volume / Target Number of Bars
```

### Threshold Adaptatif (optionnel)

```
E_k = alpha * D_k + (1 - alpha) * E_{k-1}  (EWMA des dollar values)
T_{k+1} = E_k  (le threshold s'adapte au regime de marche)
```

## Utilisation

```bash
python -m src.data_preparation.main
```

## Configuration

Parametres principaux :
- `target_num_bars` : Nombre cible de bars (defaut: 500,000)
- `calibration_fraction` : Fraction des donnees pour calibration (defaut: 0.2 = 20%)
- `adaptive` : Utiliser threshold EWMA adaptatif (defaut: False)
- `threshold_bounds` : Bornes optionnelles pour mode adaptatif

## Sortie

Colonnes du DataFrame des dollar bars :
- `bar_id` : Identifiant de la bar
- `timestamp_open/close` : Horodatages (ms)
- `datetime_open/close` : Datetime UTC
- `open, high, low, close` : OHLC
- `volume` : Volume cumule
- `cum_dollar_value` : Valeur dollar cumulee
- `vwap` : Volume Weighted Average Price
- `n_ticks` : Nombre de ticks dans la bar
- `threshold_used` : Threshold utilise
- `duration_sec` : Duree en secondes
- `log_return` : Log-return (apres ajout)

## Fichiers de Sortie

- `DOLLAR_BARS_PARQUET` : Donnees en format Parquet
- `DOLLAR_BARS_CSV` : Donnees en format CSV

## Reference

Lopez de Prado, M. (2018). Advances in Financial Machine Learning. Chapter 2: Financial Data Structures, pp. 23-30.
