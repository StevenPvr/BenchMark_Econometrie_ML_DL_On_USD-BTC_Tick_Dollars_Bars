# Data Preparation Module

Implementation des Dollar Bars selon la methodologie de De Prado (AFML Chapter 2).

## Fichiers

| Fichier | Description |
|---------|-------------|
| `preparation.py` | Algorithme Dollar Bars optimise avec Numba |
| `main.py` | Point d'entree CLI |
| `__init__.py` | Exports du module |

## Dollar Bars (De Prado)

Les Dollar Bars echantillonnent les donnees chaque fois qu'une valeur monetaire predefinie T est echangee, plutot qu'a intervalles de temps fixes.

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

Le threshold initial T_0 est calibre sur un **prefixe** des donnees :

```
T_0 = Total Dollar Volume (prefix) / Target Number of Bars (prefix)
```

Par defaut, `calibration_fraction=0.2` utilise les **premiers 20%** des ticks pour cette calibration. Cela evite d'utiliser des donnees futures pour estimer le threshold.

**Important** : Le prefixe de calibration est ensuite utilise pour generer des bars. Si vous souhaitez une separation stricte, utilisez `exclude_calibration_prefix=True` pour exclure ces bars du resultat final.

### Threshold Adaptatif (optionnel)

Apres la fermeture de la bar k avec dollar value D_k :

```
E_k = alpha * D_k + (1 - alpha) * E_{k-1}  (EWMA des dollar values)
T_{k+1} = E_k  (le threshold s'adapte au regime de marche)
```

Bornes optionnelles : `threshold_bounds=(0.5, 2.0)` limite les variations du threshold.

## Utilisation

```bash
python -m src.data_preparation.main
```

## Configuration

Les parametres sont definis dans `src/constants.py` :

| Parametre | Defaut | Description |
|-----------|--------|-------------|
| `DOLLAR_BARS_TARGET_NUM_BARS` | 500,000 | Nombre cible de bars |
| `DOLLAR_BARS_CALIBRATION_FRACTION` | 0.2 | Fraction des donnees pour calibration (20%) |
| `DOLLAR_BARS_EMA_SPAN` | 100 | Span EMA pour mode adaptatif |
| `DOLLAR_BARS_THRESHOLD_BOUNDS_DEFAULT` | (0.5, 2.0) | Bornes par defaut pour mode adaptatif |
| `DOLLAR_BARS_INCLUDE_INCOMPLETE_FINAL` | False | Inclure la bar finale incomplete |
| `DOLLAR_BARS_EXCLUDE_CALIBRATION_PREFIX` | False | Exclure les bars du prefixe de calibration |

### Parametres Fonction `compute_dollar_bars()`

- `target_num_bars` : Nombre cible de bars (obligatoire sauf si threshold est fourni)
- `threshold` : Override avec un threshold fixe (ignore target_num_bars)
- `calibration_fraction` : Fraction du prefixe pour calibration (defaut: 1.0 = tout le dataset)
- `adaptive` : Utiliser threshold EWMA adaptatif (defaut: False)
- `ema_span` : Span pour EMA adaptatif (defaut: 100)
- `threshold_bounds` : Bornes (min_mult, max_mult) pour mode adaptatif
- `include_incomplete_final` : Inclure la bar finale incomplete (defaut: False)
- `exclude_calibration_prefix` : Exclure les bars du prefixe de calibration (defaut: False)

## Prevention du Data Leakage

Le module implemente plusieurs garde-fous contre le data leakage :

1. **Calibration sur prefixe uniquement** : Le threshold T_0 est calcule uniquement sur les premiers N% des ticks (pas de lookahead)

2. **Option d'exclusion du prefixe** : `exclude_calibration_prefix=True` permet d'exclure les bars generees a partir des ticks de calibration

3. **Barre finale incomplete exclue** : Par defaut, la derniere bar qui n'a pas atteint le threshold est exclue (proprietes statistiques differentes)

4. **Validation des parametres** : Les parametres sont valides avant execution pour eviter les erreurs silencieuses

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
