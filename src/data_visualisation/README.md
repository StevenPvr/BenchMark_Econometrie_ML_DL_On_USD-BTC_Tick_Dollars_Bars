# Data Visualisation Module

Module de visualisation et d'analyse statistique des dollar bars et log-returns.

## Fichiers

| Fichier | Description |
|---------|-------------|
| `visualisation.py` | Fonctions de visualisation et tests statistiques |
| `main.py` | Point d'entree CLI |
| `__init__.py` | Exports du module |

## Analyses Disponibles (11 etapes)

### 1. Distribution des Log-Returns

- Histogramme avec estimation KDE
- Overlay distribution normale
- Q-Q Plot pour verifier la normalite
- Statistiques : moyenne, ecart-type, skewness, kurtosis

### 2. Tests de Normalite (Jarque-Bera, Shapiro-Wilk)

- **Jarque-Bera** : H0 = distribution normale (base sur skewness/kurtosis)
- **Shapiro-Wilk** : H0 = distribution normale (sous-echantillonne si n > 5000)
- Resultats exportes en JSON (`normality_tests.json`)

### 3. Tests de Stationnarite

- **ADF (Augmented Dickey-Fuller)** : H0 = serie non stationnaire
- **KPSS** : H0 = serie stationnaire
- Resultats exportes en JSON (`stationarity_tests.json`)

### 4. Analyse de Stationnarite Visuelle

- Serie temporelle avec rolling mean
- Rolling standard deviation (volatilite)
- Annotations des resultats des tests

### 5. Serie Temporelle

- Plot des log-returns dans le temps
- Volatilite realisee (rolling std)

### 6. Autocorrelation des Log-Returns

- ACF (Autocorrelation Function)
- PACF (Partial Autocorrelation Function)
- Identification des patterns AR/MA

### 7. Autocorrelation des Log-Returns² (Volatilite Clustering)

- ACF/PACF des log-returns au carre
- Detection du clustering de volatilite (effet ARCH)
- Utile pour justifier l'utilisation de modeles de volatilite

### 8. Test de Ljung-Box

- Test formel d'autocorrelation
- H0 : pas d'autocorrelation jusqu'au lag k
- Teste par defaut les lags 10, 20, 40
- Resultats exportes en JSON (`ljung_box_test.json`)

### 9. Analyse de Tendance

- **Test de Mann-Kendall** : detection tendance monotone
- Regression lineaire (pente, R2, p-value)
- Comparaison 1ere vs 2eme moitie

### 10. Visualisation de Tendance

- Plot 4 panneaux avec regression, MAs, distributions, residus

### 11. Extraction de Tendance

- Moyennes mobiles multiples (20, 50, 100, 200)
- Composante de tendance
- Residus (prix - tendance)
- Momentum (MA courte - MA longue)

## Utilisation

```bash
python -m src.data_visualisation.main
```

## Parametres

- `parquet_path` : Chemin vers les dollar bars
- `output_dir` : Repertoire de sortie des plots
- `show_plots` : Afficher les plots (defaut: True)
- `sample_fraction` : Fraction d'echantillonnage systematique (defaut: 0.2 = 20%)

**Note** : L'echantillonnage est systematique (1 barre sur N) pour preserver la structure temporelle des dollar bars.

## Sorties

Fichiers generes dans `output_dir` :

| Fichier | Description |
|---------|-------------|
| `log_returns_distribution.png` | Distribution des log-returns |
| `normality_tests.json` | Resultats Jarque-Bera/Shapiro-Wilk |
| `stationarity_tests.json` | Resultats ADF/KPSS |
| `stationarity_plot.png` | Visualisation stationnarite |
| `log_returns_time_series.png` | Serie temporelle |
| `log_returns_acf.png` | ACF/PACF log-returns |
| `log_returns_squared_acf.png` | ACF/PACF log-returns² (volatilite) |
| `ljung_box_test.json` | Resultats test Ljung-Box |
| `trend_analysis.json` | Resultats analyse tendance |
| `trend_analysis.png` | Visualisation tendance |
| `trend_extraction.png` | Extraction tendance (MAs) |

## Fonctions Exportees

```python
from src.data_visualisation import (
    compute_autocorrelation,
    compute_autocorrelation_squared,
    compute_log_returns,
    compute_trend_statistics,
    load_dollar_bars,
    mann_kendall_test,
    plot_log_returns_distribution,
    plot_log_returns_time_series,
    plot_stationarity,
    plot_trend_analysis,
    plot_trend_extraction,
    run_full_analysis,
    run_ljung_box_test,
    run_normality_tests,
    run_stationarity_tests,
)
```
