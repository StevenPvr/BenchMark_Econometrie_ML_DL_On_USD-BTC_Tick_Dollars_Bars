# Data Visualisation Module

Module de visualisation et d'analyse statistique des dollar bars et log-returns.

## Fichiers

| Fichier | Description |
|---------|-------------|
| `visualisation.py` | Fonctions de visualisation et tests statistiques |
| `main.py` | Point d'entree CLI |
| `__init__.py` | Exports du module |

## Analyses Disponibles

### 1. Distribution des Log-Returns
- Histogramme avec estimation KDE
- Overlay distribution normale
- Q-Q Plot pour verifier la normalite
- Statistiques : moyenne, ecart-type, skewness, kurtosis

### 2. Tests de Stationnarite
- **ADF (Augmented Dickey-Fuller)** : H0 = serie non stationnaire
- **KPSS** : H0 = serie stationnaire
- Resultats exportes en JSON

### 3. Analyse de Stationnarite Visuelle
- Serie temporelle avec rolling mean
- Rolling standard deviation (volatilite)
- Annotations des resultats des tests

### 4. Serie Temporelle
- Plot des log-returns dans le temps
- Volatilite realisee (rolling std)

### 5. Autocorrelation
- ACF (Autocorrelation Function)
- PACF (Partial Autocorrelation Function)
- Identification des patterns AR/MA

### 6. Analyse de Tendance
- **Test de Mann-Kendall** : detection tendance monotone
- Regression lineaire (pente, R2, p-value)
- Comparaison 1ere vs 2eme moitie

### 7. Extraction de Tendance
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
- `sample_fraction` : Fraction d'echantillonnage (defaut: 0.2 = 20%)

## Sorties

Fichiers generes dans `output_dir` :
- `log_returns_distribution.png` : Distribution des log-returns
- `stationarity_tests.json` : Resultats ADF/KPSS
- `stationarity_plot.png` : Visualisation stationnarite
- `log_returns_time_series.png` : Serie temporelle
- `log_returns_acf.png` : ACF/PACF
- `trend_analysis.json` : Resultats analyse tendance
- `trend_analysis.png` : Visualisation tendance
- `trend_extraction.png` : Extraction tendance (MAs)

## Utilitaires GARCH

Le module inclut des utilitaires pour la visualisation GARCH :
- `create_figure_canvas()` : Creation de figures
- `save_canvas()` : Sauvegarde de figures
- `add_zero_line()` : Ajout ligne horizontale
- `plot_histogram_with_normal_overlay()` : Histogramme avec normale
