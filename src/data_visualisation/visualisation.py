"""Visualisation module for dollar bars log-returns analysis."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Sequence, cast

import matplotlib.pyplot as plt # type: ignore
import numpy as np
import pandas as pd  # type: ignore
from matplotlib.axes import Axes
from matplotlib.figure import Figure # type: ignore
from statsmodels.tsa.stattools import adfuller, kpss  # type: ignore
from scipy import stats  # type: ignore
from scipy.stats import linregress  # type: ignore

from ..constants import CLOSE_COLUMN, LOG_RETURN_COLUMN # type: ignore
from ..path import DOLLAR_BARS_PARQUET


def load_dollar_bars(parquet_path: Path = DOLLAR_BARS_PARQUET) -> pd.DataFrame:
    """
    Charge les dollar bars depuis le fichier parquet.

    Parameters
    ----------
    parquet_path : Path
        Chemin vers le fichier parquet des dollar bars.

    Returns
    -------
    pd.DataFrame
        DataFrame contenant les dollar bars.
    """
    if not parquet_path.exists():
        raise FileNotFoundError(f"Dollar bars file not found: {parquet_path}")
    return pd.read_parquet(parquet_path)


def compute_log_returns(df: pd.DataFrame, price_col: str = CLOSE_COLUMN) -> pd.Series:
    """
    Calcule les log-returns a partir des prix de cloture.

    log_return_t = ln(P_t / P_{t-1}) = ln(P_t) - ln(P_{t-1})

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant les dollar bars.
    price_col : str, default=CLOSE_COLUMN ("close")
        Nom de la colonne de prix.

    Returns
    -------
    pd.Series
        Serie des log-returns (premiere valeur = NaN).
    """
    prices = df[price_col]
    log_returns = np.log(prices / prices.shift(1))
    return log_returns


def plot_log_returns_distribution(
    log_returns: pd.Series,
    output_dir: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 5),
) -> Figure:
    """
    Plot la distribution des log-returns.

    Parameters
    ----------
    log_returns : pd.Series
        Serie des log-returns.
    output_dir : Path, optional
        Repertoire ou sauvegarder le plot.
    figsize : Tuple[int, int]
        Taille de la figure.

    Returns
    -------
    Figure
        Figure matplotlib.
    """
    data = log_returns.dropna()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Histogram avec KDE
    ax1 = axes[0]
    ax1.hist(data, bins=100, alpha=0.7, color="steelblue", edgecolor="white", density=True)

    # KDE
    kde = stats.gaussian_kde(data)
    x_range = np.linspace(data.min(), data.max(), 200)
    ax1.plot(x_range, kde(x_range), color="red", linewidth=2, label="KDE")

    # Normal distribution overlay
    mu, std = data.mean(), data.std()
    normal_pdf = stats.norm.pdf(x_range, mu, std)
    ax1.plot(x_range, normal_pdf, color="green", linewidth=2, linestyle="--", label="Normal")

    ax1.axvline(mu, color="red", linestyle=":", linewidth=1.5, alpha=0.7)
    ax1.set_title("Distribution des Log-Returns", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Log-Return")
    ax1.set_ylabel("Densite")
    ax1.legend()

    # QQ-plot
    ax2 = axes[1]
    stats.probplot(data, dist="norm", plot=ax2)
    ax2.set_title("Q-Q Plot (Normal)", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # Stats
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    fig.suptitle(
        f"Log-Returns: Mean={mu:.6f}, Std={std:.6f}, Skew={skewness:.3f}, Kurt={kurtosis:.3f}",
        fontsize=10, y=1.02
    )

    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / "log_returns_distribution.png", dpi=150, bbox_inches="tight")
        print(f"Distribution plot saved: {output_dir / 'log_returns_distribution.png'}")

    return fig


def run_stationarity_tests(
    log_returns: pd.Series,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Effectue les tests de stationnarite ADF et KPSS sur les log-returns.

    Parameters
    ----------
    log_returns : pd.Series
        Serie des log-returns.
    output_dir : Path, optional
        Repertoire ou sauvegarder les resultats.

    Returns
    -------
    Dict[str, Any]
        Dictionnaire avec les resultats des tests.
    """
    import json

    series = log_returns.dropna().values

    results: Dict[str, Any] = {}

    # ADF Test (H0: serie non stationnaire)
    try:
        adf_result = adfuller(series, autolag="AIC")
        adf_stat = float(adf_result[0])
        adf_pvalue = float(adf_result[1])
        adf_critical = adf_result[4] if len(adf_result) > 4 else {}  # type: ignore
        adf_conclusion = "Stationnaire" if adf_pvalue < 0.05 else "Non-stationnaire"
        results["ADF"] = {
            "statistic": adf_stat,
            "p_value": adf_pvalue,
            "conclusion": adf_conclusion,
            "critical_values": {
                "1%": float(adf_critical.get("1%", 0)) if isinstance(adf_critical, dict) else None,
                "5%": float(adf_critical.get("5%", 0)) if isinstance(adf_critical, dict) else None,
                "10%": float(adf_critical.get("10%", 0)) if isinstance(adf_critical, dict) else None,
            },
            "hypothesis": "H0: serie non stationnaire (p < 0.05 => rejeter H0 => stationnaire)",
        }
    except Exception as e:
        print(f"ADF test failed: {e}")
        results["ADF"] = {"error": str(e)}

    # KPSS Test (H0: serie stationnaire)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*p-value.*")
            kpss_result = kpss(series, regression="c", nlags="auto")
        kpss_stat = float(kpss_result[0])
        kpss_pvalue = float(kpss_result[1])
        kpss_critical = kpss_result[3]
        kpss_conclusion = "Stationnaire" if kpss_pvalue > 0.05 else "Non-stationnaire"
        results["KPSS"] = {
            "statistic": kpss_stat,
            "p_value": kpss_pvalue,
            "conclusion": kpss_conclusion,
            "critical_values": {
                "1%": float(kpss_critical.get("1%", 0)) if isinstance(kpss_critical, dict) else None,
                "5%": float(kpss_critical.get("5%", 0)) if isinstance(kpss_critical, dict) else None,
                "10%": float(kpss_critical.get("10%", 0)) if isinstance(kpss_critical, dict) else None,
            },
            "hypothesis": "H0: serie stationnaire (p > 0.05 => ne pas rejeter H0 => stationnaire)",
        }
    except Exception as e:
        print(f"KPSS test failed: {e}")
        results["KPSS"] = {"error": str(e)}

    print("\n" + "=" * 80)
    print("TESTS DE STATIONNARITE - LOG-RETURNS")
    print("=" * 80)
    for test_name, test_results in results.items():
        if "error" not in test_results:
            print(f"\n{test_name}:")
            print(f"  Statistic: {test_results['statistic']:.6f}")
            print(f"  P-value: {test_results['p_value']:.6f}")
            print(f"  Conclusion: {test_results['conclusion']}")
    print("=" * 80 + "\n")

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "stationarity_tests.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"Stationarity tests saved: {output_dir / 'stationarity_tests.json'}")

    return results


def plot_stationarity(
    log_returns: pd.Series,
    stationarity_results: Dict[str, Any],
    output_dir: Optional[Path] = None,
    window: int = 50,
    figsize: Tuple[int, int] = (14, 8),
) -> Figure:
    """
    Plot de stationnarite: rolling mean et rolling std des log-returns.

    Parameters
    ----------
    log_returns : pd.Series
        Serie des log-returns.
    stationarity_results : Dict[str, Any]
        Resultats des tests de stationnarite.
    output_dir : Path, optional
        Repertoire ou sauvegarder le plot.
    window : int
        Fenetre pour les statistiques rolling.
    figsize : Tuple[int, int]
        Taille de la figure.

    Returns
    -------
    Figure
        Figure matplotlib.
    """
    data = log_returns.dropna()

    # Calcul des statistiques rolling
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()

    fig, axes = plt.subplots(2, 1, figsize=figsize)

    # Plot 1: Serie + Rolling Mean
    ax1 = axes[0]
    ax1.plot(data, linewidth=0.5, color="steelblue", alpha=0.6, label="Log-Returns")
    ax1.plot(rolling_mean, linewidth=2, color="red", label=f"Rolling Mean ({window})")
    ax1.axhline(data.mean(), color="green", linestyle="--", linewidth=1.5, label=f"Mean globale: {data.mean():.6f}")
    ax1.set_ylabel("Log-Return", fontsize=10)
    ax1.set_title("Log-Returns et Rolling Mean", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Rolling Std
    ax2 = axes[1]
    ax2.plot(rolling_std.values, linewidth=1.5, color="orange", label=f"Rolling Std ({window})")
    ax2.axhline(data.std(), color="green", linestyle="--", linewidth=1.5, label=f"Std globale: {data.std():.6f}")
    ax2.fill_between(range(len(rolling_std)), 0, rolling_std.values, alpha=0.3, color="orange")
    ax2.set_xlabel("Index", fontsize=10)
    ax2.set_ylabel("Ecart-type", fontsize=10)
    ax2.set_title("Rolling Standard Deviation (Volatilite)", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Ajouter les resultats des tests en annotation
    adf_result = stationarity_results.get("ADF", {})
    kpss_result = stationarity_results.get("KPSS", {})

    test_text = "Tests de Stationnarite:\n"
    if "conclusion" in adf_result:
        test_text += f"ADF: {adf_result['conclusion']} (p={adf_result['p_value']:.4f})\n"
    if "conclusion" in kpss_result:
        test_text += f"KPSS: {kpss_result['conclusion']} (p={kpss_result['p_value']:.4f})"

    fig.text(0.02, 0.98, test_text, transform=fig.transFigure, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / "stationarity_plot.png", dpi=150, bbox_inches="tight")
        print(f"Stationarity plot saved: {output_dir / 'stationarity_plot.png'}")

    return fig


def plot_log_returns_time_series(
    df: pd.DataFrame,
    log_returns: pd.Series,
    datetime_col: str = "datetime_close",
    output_dir: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 8),
) -> Figure:
    """
    Plot la serie temporelle des log-returns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame des dollar bars (pour datetime).
    log_returns : pd.Series
        Serie des log-returns.
    datetime_col : str
        Nom de la colonne datetime.
    output_dir : Path, optional
        Repertoire ou sauvegarder le plot.
    figsize : Tuple[int, int]
        Taille de la figure.

    Returns
    -------
    Figure
        Figure matplotlib.
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Datetime
    if datetime_col in df.columns:
        x = pd.to_datetime(df[datetime_col])
    else:
        x = df.index

    # Log-returns
    ax1 = axes[0]
    ax1.plot(x, log_returns, linewidth=0.5, color="steelblue", alpha=0.8)
    ax1.axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax1.set_ylabel("Log-Return", fontsize=10)
    ax1.set_title("Serie Temporelle des Log-Returns", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Volatilite (rolling std)
    ax2 = axes[1]
    rolling_vol = log_returns.rolling(window=20).std()
    ax2.plot(x, rolling_vol, linewidth=1, color="orange", alpha=0.8)
    ax2.fill_between(x, 0, rolling_vol, alpha=0.3, color="orange")
    ax2.set_ylabel("Volatilite (Std 20)", fontsize=10)
    ax2.set_xlabel("Date", fontsize=10)
    ax2.set_title("Volatilite Realisee (Rolling Std)", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / "log_returns_time_series.png", dpi=150, bbox_inches="tight")
        print(f"Time series plot saved: {output_dir / 'log_returns_time_series.png'}")

    return fig


def compute_autocorrelation(
    log_returns: pd.Series,
    output_dir: Optional[Path] = None,
    max_lags: int = 40,
    figsize: Tuple[int, int] = (14, 5),
) -> Figure:
    """
    Plot l'autocorrelation des log-returns et des log-returns au carre.

    Parameters
    ----------
    log_returns : pd.Series
        Serie des log-returns.
    output_dir : Path, optional
        Repertoire ou sauvegarder le plot.
    max_lags : int
        Nombre maximum de lags.
    figsize : Tuple[int, int]
        Taille de la figure.

    Returns
    -------
    Figure
        Figure matplotlib.
    """
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # type: ignore

    data = log_returns.dropna()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ACF des log-returns (sans lag 0)
    plot_acf(data, lags=max_lags, ax=axes[0], alpha=0.05, zero=False)
    axes[0].set_title("ACF - Log-Returns", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Lag")
    axes[0].set_ylabel("Autocorrelation")

    # PACF des log-returns (sans lag 0)
    plot_pacf(data, lags=max_lags, ax=axes[1], alpha=0.05, zero=False)
    axes[1].set_title("PACF - Log-Returns", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Lag")
    axes[1].set_ylabel("Partial Autocorrelation")

    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / "log_returns_acf.png", dpi=150, bbox_inches="tight")
        print(f"ACF plot saved: {output_dir / 'log_returns_acf.png'}")

    return fig


def compute_autocorrelation_squared(
    log_returns: pd.Series,
    output_dir: Optional[Path] = None,
    max_lags: int = 40,
    figsize: Tuple[int, int] = (14, 5),
) -> Figure:
    """
    Plot l'autocorrelation des log-returns au carre (volatilite clustering).

    L'autocorrelation des log-returns au carre permet de detecter le
    clustering de volatilite, caracteristique des series financieres.

    Parameters
    ----------
    log_returns : pd.Series
        Serie des log-returns.
    output_dir : Path, optional
        Repertoire ou sauvegarder le plot.
    max_lags : int
        Nombre maximum de lags.
    figsize : Tuple[int, int]
        Taille de la figure.

    Returns
    -------
    Figure
        Figure matplotlib.
    """
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # type: ignore

    data = log_returns.dropna()
    squared = data ** 2

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ACF des log-returns au carre (sans lag 0)
    plot_acf(squared, lags=max_lags, ax=axes[0], alpha=0.05, zero=False)
    axes[0].set_title("ACF - Log-Returns² (Volatilite)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Lag")
    axes[0].set_ylabel("Autocorrelation")

    # PACF des log-returns au carre (sans lag 0)
    plot_pacf(squared, lags=max_lags, ax=axes[1], alpha=0.05, zero=False)
    axes[1].set_title("PACF - Log-Returns² (Volatilite)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Lag")
    axes[1].set_ylabel("Partial Autocorrelation")

    fig.suptitle(
        "Autocorrelation des Log-Returns² (detection du clustering de volatilite)",
        fontsize=11, y=1.02
    )

    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / "log_returns_squared_acf.png", dpi=150, bbox_inches="tight")
        print(f"ACF squared plot saved: {output_dir / 'log_returns_squared_acf.png'}")

    return fig


def run_ljung_box_test(
    series: pd.Series,
    lags: Optional[List[int]] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Test de Ljung-Box pour l'autocorrelation.

    H0: Pas d'autocorrelation jusqu'au lag k
    H1: Autocorrelation presente

    Parameters
    ----------
    series : pd.Series
        Serie temporelle (log-returns ou residus).
    lags : List[int], optional
        Lags a tester. Par defaut [10, 20, 40].
    output_dir : Path, optional
        Repertoire ou sauvegarder les resultats.

    Returns
    -------
    Dict[str, Any]
        Resultats du test pour chaque lag.
    """
    import json
    from statsmodels.stats.diagnostic import acorr_ljungbox  # type: ignore

    if lags is None:
        lags = [10, 20, 40]

    data = series.dropna()

    results: Dict[str, Any] = {
        "test_name": "Ljung-Box",
        "hypothesis": "H0: Pas d'autocorrelation jusqu'au lag k (p < 0.05 => rejeter H0)",
        "lags_tested": {},
    }

    try:
        lb_results = acorr_ljungbox(data, lags=lags, return_df=True)

        for lag in lags:
            if lag in lb_results.index:
                lb_stat = float(lb_results.loc[lag, "lb_stat"])
                lb_pvalue = float(lb_results.loc[lag, "lb_pvalue"])
                conclusion = "Autocorrelation significative" if lb_pvalue < 0.05 else "Pas d'autocorrelation"
                results["lags_tested"][str(lag)] = {
                    "statistic": lb_stat,
                    "p_value": lb_pvalue,
                    "conclusion": conclusion,
                }
    except Exception as e:
        results["error"] = str(e)

    # Affichage
    print("\n" + "=" * 60)
    print("TEST DE LJUNG-BOX (Autocorrelation)")
    print("=" * 60)
    if "error" not in results:
        for lag_str, lag_results in results["lags_tested"].items():
            print(f"  Lag {lag_str}: stat={lag_results['statistic']:.4f}, p={lag_results['p_value']:.4f} -> {lag_results['conclusion']}")
    else:
        print(f"  Erreur: {results['error']}")
    print("=" * 60 + "\n")

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "ljung_box_test.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"Ljung-Box test saved: {output_dir / 'ljung_box_test.json'}")

    return results


def run_normality_tests(
    series: pd.Series,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Tests de normalite (Jarque-Bera et Shapiro-Wilk).

    Jarque-Bera: H0 = Distribution normale (basee sur skewness et kurtosis)
    Shapiro-Wilk: H0 = Distribution normale (pour n < 5000)

    Parameters
    ----------
    series : pd.Series
        Serie temporelle (log-returns).
    output_dir : Path, optional
        Repertoire ou sauvegarder les resultats.

    Returns
    -------
    Dict[str, Any]
        Resultats des tests de normalite.
    """
    import json
    from scipy.stats import jarque_bera, shapiro  # type: ignore

    data = series.dropna().values

    results: Dict[str, Any] = {
        "n_observations": len(data),
    }

    # Test de Jarque-Bera
    try:
        jb_stat, jb_pvalue = jarque_bera(data)
        jb_conclusion = "Non normale" if jb_pvalue < 0.05 else "Compatible avec normale"
        results["Jarque_Bera"] = {
            "statistic": float(jb_stat),
            "p_value": float(jb_pvalue),
            "conclusion": jb_conclusion,
            "hypothesis": "H0: Distribution normale (p < 0.05 => rejeter H0 => non normale)",
        }
    except Exception as e:
        results["Jarque_Bera"] = {"error": str(e)}

    # Test de Shapiro-Wilk (limite a 5000 observations)
    try:
        if len(data) <= 5000:
            sw_stat, sw_pvalue = shapiro(data)
            sw_conclusion = "Non normale" if sw_pvalue < 0.05 else "Compatible avec normale"
            results["Shapiro_Wilk"] = {
                "statistic": float(sw_stat),
                "p_value": float(sw_pvalue),
                "conclusion": sw_conclusion,
                "hypothesis": "H0: Distribution normale (p < 0.05 => rejeter H0 => non normale)",
            }
        else:
            # Sous-echantillonnage pour Shapiro-Wilk
            np.random.seed(42)
            sample = np.random.choice(data, size=5000, replace=False)
            sw_stat, sw_pvalue = shapiro(sample)
            sw_conclusion = "Non normale" if sw_pvalue < 0.05 else "Compatible avec normale"
            results["Shapiro_Wilk"] = {
                "statistic": float(sw_stat),
                "p_value": float(sw_pvalue),
                "conclusion": sw_conclusion,
                "hypothesis": "H0: Distribution normale (test sur 5000 obs. aleatoires)",
                "note": "Sous-echantillonne a 5000 observations (limite Shapiro-Wilk)",
            }
    except Exception as e:
        results["Shapiro_Wilk"] = {"error": str(e)}

    # Affichage
    print("\n" + "=" * 60)
    print("TESTS DE NORMALITE")
    print("=" * 60)
    for test_name in ["Jarque_Bera", "Shapiro_Wilk"]:
        test_res = results.get(test_name, {})
        if "error" not in test_res and "statistic" in test_res:
            print(f"  {test_name}: stat={test_res['statistic']:.4f}, p={test_res['p_value']:.6f} -> {test_res['conclusion']}")
        elif "error" in test_res:
            print(f"  {test_name}: Erreur - {test_res['error']}")
    print("=" * 60 + "\n")

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "normality_tests.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"Normality tests saved: {output_dir / 'normality_tests.json'}")

    return results


def mann_kendall_test(series: np.ndarray, max_samples: int = 5000) -> Dict[str, Any]:
    """
    Test de Mann-Kendall pour detecter une tendance monotone.

    H0: Pas de tendance monotone
    H1: Tendance monotone presente

    Parameters
    ----------
    series : np.ndarray
        Serie temporelle.
    max_samples : int
        Nombre maximum d'echantillons pour le calcul (sous-echantillonnage
        si n > max_samples pour eviter O(n²) sur de gros datasets).

    Returns
    -------
    Dict[str, Any]
        Resultats du test (statistic, p_value, trend).
    """
    n = len(series)

    # Sous-echantillonnage si trop de donnees (le test O(n²) devient prohibitif)
    if n > max_samples:
        indices = np.linspace(0, n - 1, max_samples, dtype=int)
        series = series[indices]
        n = max_samples

    # Calcul vectorise de la statistique S
    # Pour chaque paire (i, j) avec i < j, calculer sign(series[j] - series[i])
    # Utilise broadcasting: series[np.newaxis, :] - series[:, np.newaxis] donne
    # une matrice ou M[i,j] = series[j] - series[i]
    # On ne garde que la partie triangulaire superieure (i < j)
    diff_matrix = series[np.newaxis, :] - series[:, np.newaxis]
    s = int(np.sum(np.sign(diff_matrix[np.triu_indices(n, k=1)])))

    # Variance de S
    var_s = n * (n - 1) * (2 * n + 5) / 18

    # Statistique Z normalisee
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0

    # P-value (test bilateral)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    # Interpretation de la tendance
    if p_value < 0.05:
        if z > 0:
            trend = "Tendance croissante significative"
        else:
            trend = "Tendance decroissante significative"
    else:
        trend = "Pas de tendance significative"

    return {
        "statistic_S": s,
        "statistic_Z": z,
        "p_value": p_value,
        "trend": trend,
        "hypothesis": "H0: Pas de tendance monotone (p < 0.05 => rejeter H0)",
    }


def compute_trend_statistics(
    series: pd.Series,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Calcule les statistiques de tendance pour une serie temporelle.

    Inclut:
    - Test de Mann-Kendall (tendance monotone)
    - Regression lineaire (pente, R²)
    - Statistiques descriptives de la tendance

    Parameters
    ----------
    series : pd.Series
        Serie temporelle (ex: prix ou log-returns cumules).
    output_dir : Path, optional
        Repertoire ou sauvegarder les resultats.

    Returns
    -------
    Dict[str, Any]
        Dictionnaire avec les resultats des tests de tendance.
    """
    import json

    data = np.asarray(series.dropna().values)
    n = len(data)
    x = np.arange(n)

    results: Dict[str, Any] = {}

    # 1. Test de Mann-Kendall
    mk_result = mann_kendall_test(data)
    results["Mann_Kendall"] = mk_result

    # 2. Regression lineaire
    lr_result = cast(Any, linregress(x, data))
    slope = lr_result.slope
    intercept = lr_result.intercept
    r_value = lr_result.rvalue
    p_value = lr_result.pvalue
    std_err = lr_result.stderr
    r_squared = r_value ** 2

    # Tendance annualisee (en supposant des barres quotidiennes)
    trend_direction = "Haussiere" if slope > 0 else "Baissiere" if slope < 0 else "Neutre"

    results["Linear_Regression"] = {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "p_value": p_value,
        "std_error": std_err,
        "trend_direction": trend_direction,
        "slope_significance": "Significatif" if p_value < 0.05 else "Non significatif",
    }

    # 3. Statistiques de changement
    first_half = np.asarray(data[:n // 2])
    second_half = np.asarray(data[n // 2:])

    results["Change_Statistics"] = {
        "mean_first_half": float(first_half.mean()),
        "mean_second_half": float(second_half.mean()),
        "mean_change": float(second_half.mean() - first_half.mean()),
        "std_first_half": float(first_half.std()),
        "std_second_half": float(second_half.std()),
        "std_change": float(second_half.std() - first_half.std()),
    }

    # Affichage
    print("\n" + "=" * 80)
    print("ANALYSE DE TENDANCE")
    print("=" * 80)

    print("\n[Mann-Kendall Test]")
    print(f"  Statistique S: {mk_result['statistic_S']}")
    print(f"  Statistique Z: {mk_result['statistic_Z']:.4f}")
    print(f"  P-value: {mk_result['p_value']:.6f}")
    print(f"  Conclusion: {mk_result['trend']}")

    print("\n[Regression Lineaire]")
    print(f"  Pente: {slope:.8f}")
    print(f"  R²: {r_squared:.6f}")
    print(f"  P-value: {p_value:.6f}")
    print(f"  Direction: {trend_direction}")
    print(f"  Significativite: {results['Linear_Regression']['slope_significance']}")

    print("\n[Changement entre les deux moities]")
    print(f"  Moyenne 1ere moitie: {results['Change_Statistics']['mean_first_half']:.6f}")
    print(f"  Moyenne 2eme moitie: {results['Change_Statistics']['mean_second_half']:.6f}")
    print(f"  Changement de moyenne: {results['Change_Statistics']['mean_change']:.6f}")

    print("=" * 80 + "\n")

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Convertir pour JSON
        json_results = {
            "Mann_Kendall": mk_result,
            "Linear_Regression": {
                k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in results["Linear_Regression"].items()
            },
            "Change_Statistics": results["Change_Statistics"],
        }
        with open(output_dir / "trend_analysis.json", "w") as f:
            json.dump(json_results, f, indent=2)
        print(f"Trend analysis saved: {output_dir / 'trend_analysis.json'}")

    return results


def plot_trend_extraction(
    series: pd.Series,
    windows: List[int] = [20, 50, 100, 200],
    output_dir: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 12),
) -> Figure:
    """
    Extrait et visualise la tendance des dollar bars via moyennes mobiles.

    Pour les dollar bars (echantillonnage par volume), la decomposition
    saisonniere classique n'est pas appropriee. On utilise plutot
    des moyennes mobiles pour extraire la tendance.

    Parameters
    ----------
    series : pd.Series
        Serie de prix des dollar bars.
    windows : List[int]
        Fenetres pour les moyennes mobiles.
    output_dir : Path, optional
        Repertoire ou sauvegarder le plot.
    figsize : Tuple[int, int]
        Taille de la figure.

    Returns
    -------
    Figure
        Figure matplotlib.
    """
    data = series.dropna()
    data_arr = np.asarray(data.values)
    n = len(data_arr)

    fig, axes = plt.subplots(4, 1, figsize=figsize)

    # Plot 1: Serie originale avec tendance longue (MA la plus longue)
    ax1 = axes[0]
    ax1.plot(data_arr, linewidth=0.5, color="steelblue", alpha=0.7, label="Prix")
    long_window = max(w for w in windows if w < n) if any(w < n for w in windows) else 50
    ma_long = cast(pd.Series, pd.Series(data_arr).rolling(window=long_window).mean())
    ax1.plot(ma_long.values, linewidth=2, color="red", label=f"Tendance (MA{long_window})")
    ax1.set_title("Dollar Bars - Prix et Tendance", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Prix")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Composante de tendance extraite (difference entre MAs)
    ax2 = axes[1]
    short_window = min(windows)
    ma_short = cast(pd.Series, pd.Series(data_arr).rolling(window=short_window).mean())
    trend_component = ma_long - ma_long.iloc[long_window - 1]  # Tendance relative
    ax2.plot(trend_component.values, linewidth=1.5, color="red")
    ax2.axhline(0, color="black", linestyle="--", linewidth=1)
    ax2.set_title(f"Composante de Tendance (MA{long_window} - depart)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Tendance relative")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Residus (prix - tendance)
    ax3 = axes[2]
    residuals = data_arr - ma_long.values
    ax3.plot(residuals, linewidth=0.8, color="orange")
    ax3.axhline(0, color="black", linestyle="--", linewidth=1)
    ax3.fill_between(range(n), 0, residuals, alpha=0.3, color="orange",
                     where=~np.isnan(residuals))
    ax3.set_title("Residus (Prix - Tendance)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Residu")
    ax3.grid(True, alpha=0.3)

    # Stats des residus
    valid_res = residuals[~np.isnan(residuals)]
    res_std = np.std(valid_res)
    res_mean = np.mean(valid_res)
    ax3.text(0.02, 0.95, f"Mean={res_mean:.4f}, Std={res_std:.4f}",
             transform=ax3.transAxes, fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 4: Momentum (difference entre MA courte et MA longue)
    ax4 = axes[3]
    momentum = cast(pd.Series, ma_short - ma_long)
    ax4.plot(momentum.values, linewidth=1.2, color="purple")
    ax4.axhline(0, color="black", linestyle="--", linewidth=1)
    momentum_values = cast(np.ndarray, momentum.values)
    ax4.fill_between(range(n), 0, momentum_values, alpha=0.3,
                     where=(momentum_values > 0) & ~np.isnan(momentum_values), color="green")
    ax4.fill_between(range(n), 0, momentum_values, alpha=0.3,
                     where=(momentum_values < 0) & ~np.isnan(momentum_values), color="red")
    ax4.set_title(f"Momentum (MA{short_window} - MA{long_window})", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Index (Dollar Bars)")
    ax4.set_ylabel("Momentum")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / "trend_extraction.png", dpi=150, bbox_inches="tight")
        print(f"Trend extraction plot saved: {output_dir / 'trend_extraction.png'}")

    return fig


def plot_trend_analysis(
    series: pd.Series,
    trend_results: Dict[str, Any],
    series_name: str = "Dollar Bars",
    output_dir: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 10),
) -> Figure:
    """
    Plot complet de l'analyse de tendance avec regression lineaire et moyennes mobiles.

    Adapte pour les dollar bars (echantillonnage par volume en dollars).

    Parameters
    ----------
    series : pd.Series
        Serie de prix des dollar bars.
    trend_results : Dict[str, Any]
        Resultats de compute_trend_statistics.
    series_name : str
        Nom de la serie pour le titre.
    output_dir : Path, optional
        Repertoire ou sauvegarder le plot.
    figsize : Tuple[int, int]
        Taille de la figure.

    Returns
    -------
    Figure
        Figure matplotlib.
    """
    data = series.dropna()
    n = len(data)
    x = np.arange(n)

    # Extraire les parametres de regression
    lr = trend_results.get("Linear_Regression", {})
    slope = lr.get("slope", 0)
    intercept = lr.get("intercept", 0)
    r_squared = lr.get("r_squared", 0)

    mk = trend_results.get("Mann_Kendall", {})

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Plot 1: Serie avec regression lineaire
    ax1 = axes[0, 0]
    ax1.plot(data.values, linewidth=0.8, color="steelblue", alpha=0.7, label=series_name)
    ax1.plot(x, slope * x + intercept, color="red", linewidth=2,
             linestyle="--", label=f"Tendance (pente={slope:.2e})")
    ax1.set_title(f"{series_name} avec Regression Lineaire", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Index (Dollar Bars)")
    ax1.set_ylabel("Prix")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Annotation R²
    ax1.text(0.95, 0.95, f"R² = {r_squared:.4f}",
             transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Moyennes mobiles multiples
    ax2 = axes[0, 1]
    data_arr = np.asarray(data.values)
    ax2.plot(data_arr, linewidth=0.5, color="steelblue", alpha=0.5, label="Original")

    windows = [20, 50, 100]
    colors = ["orange", "red", "purple"]
    for window, color in zip(windows, colors):
        if window < n:
            ma = cast(pd.Series, pd.Series(data_arr).rolling(window=window).mean())
            ax2.plot(ma.values, linewidth=1.5, color=color, label=f"MA({window})")

    ax2.set_title("Moyennes Mobiles", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Index")
    ax2.set_ylabel("Valeur")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Distribution premiere vs deuxieme moitie
    ax3 = axes[1, 0]
    first_half = data_arr[:n // 2]
    second_half = data_arr[n // 2:]

    ax3.hist(first_half, bins=40, alpha=0.6, color="blue", label="1ere moitie", density=True)
    ax3.hist(second_half, bins=40, alpha=0.6, color="red", label="2eme moitie", density=True)
    ax3.axvline(first_half.mean(), color="blue", linestyle="--", linewidth=2)
    ax3.axvline(second_half.mean(), color="red", linestyle="--", linewidth=2)
    ax3.set_title("Distribution: 1ere vs 2eme moitie", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Valeur")
    ax3.set_ylabel("Densite")
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Residus de la tendance
    ax4 = axes[1, 1]
    trend_line = slope * x + intercept
    residuals = data.values - trend_line
    ax4.plot(residuals, linewidth=0.8, color="orange")
    ax4.axhline(0, color="black", linestyle="--", linewidth=1)
    ax4.fill_between(x, 0, residuals, alpha=0.3, color="orange")
    ax4.set_title("Residus apres suppression de la tendance", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Index")
    ax4.set_ylabel("Residu")
    ax4.grid(True, alpha=0.3)

    # Stats residus
    res_std = np.std(residuals)
    ax4.text(0.95, 0.95, f"Std residus = {res_std:.4f}",
             transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Titre global avec resultats Mann-Kendall
    mk_trend = mk.get("trend", "N/A")
    mk_pvalue = mk.get("p_value", 0)
    fig.suptitle(
        f"Analyse de Tendance - Mann-Kendall: {mk_trend} (p={mk_pvalue:.4f})",
        fontsize=13, fontweight="bold", y=1.02
    )

    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / "trend_analysis.png", dpi=150, bbox_inches="tight")
        print(f"Trend analysis plot saved: {output_dir / 'trend_analysis.png'}")

    return fig


def run_full_analysis(
    parquet_path: Path = DOLLAR_BARS_PARQUET,
    output_dir: Optional[Path] = None,
    show_plots: bool = True,
    sample_fraction: float = 1.0,  # New parameter for sampling
) -> Dict[str, Any]:
    """
    Execute l'analyse complete des log-returns des dollar bars.

    Parameters
    ----------
    parquet_path : Path
        Chemin vers le fichier parquet des dollar bars.
    output_dir : Path, optional
        Repertoire ou sauvegarder les resultats.
    show_plots : bool
        Si True, affiche les plots.
    sample_fraction : float
        Fraction du dataset a utiliser pour l'analyse (0.0 a 1.0).

    Returns
    -------
    dict
        Dictionnaire contenant les resultats de l'analyse.
    """
    print("=" * 80)
    print("ANALYSE DES LOG-RETURNS DES DOLLAR BARS")
    print("=" * 80)

    # Charger les donnees
    df = load_dollar_bars(parquet_path)
    print(f"\nDonnees chargees: {len(df)} barres")
    print(f"Periode: {df['datetime_close'].min()} a {df['datetime_close'].max()}")

    # Appliquer l'echantillonnage systematique si necessaire
    # (preserve la structure temporelle des dollar bars)
    if sample_fraction < 1.0 and not df.empty:
        original_len = len(df)
        step = max(1, int(1 / sample_fraction))
        df = df.iloc[::step].reset_index(drop=True)
        print(f"Echantillonnage systematique applique: {len(df)} barres (1/{step} = {len(df)/original_len:.1%} de l'original)")
        if df.empty:
            raise ValueError("Dataset vide apres echantillonnage. Essayez d'augmenter sample_fraction.")

    # Utiliser les log-returns existants (naturels)
    print("\n[0/8] Chargement des log-returns (naturels)...")
    if LOG_RETURN_COLUMN in df.columns:
        log_returns = df[LOG_RETURN_COLUMN]
        print("Log-returns charges depuis la colonne existante (naturels)")
    else:
        # Fallback: calculer si la colonne n'existe pas
        log_returns = compute_log_returns(df, price_col=CLOSE_COLUMN)
        df[LOG_RETURN_COLUMN] = log_returns
        print("Log-returns calcules (naturels) car colonne absente")

    n_valid = log_returns.dropna().shape[0]
    print(f"Log-returns (naturels): {n_valid} observations")
    print(f"  Mean: {log_returns.mean():.6f}")
    print(f"  Std:  {log_returns.std():.6f}")
    print(f"  Min:  {log_returns.min():.6f}")
    print(f"  Max:  {log_returns.max():.6f}")

    results: Dict[str, Any] = {
        "n_bars": len(df),
        "n_log_returns": n_valid,
        "log_returns": log_returns,
    }

    # 1. Distribution
    print("\n[1/8] Generation du plot de distribution...")
    fig_dist = plot_log_returns_distribution(log_returns, output_dir=output_dir)
    results["fig_distribution"] = fig_dist

    # 2. Tests de normalite (Jarque-Bera, Shapiro-Wilk)
    print("\n[2/8] Execution des tests de normalite...")
    normality_results = run_normality_tests(log_returns, output_dir=output_dir)
    results["normality_tests"] = normality_results

    # 3. Tests de stationnarite
    print("\n[3/8] Execution des tests de stationnarite...")
    stationarity_results = run_stationarity_tests(log_returns, output_dir=output_dir)
    results["stationarity_tests"] = stationarity_results

    # 4. Plot de stationnarite
    print("\n[4/8] Generation du plot de stationnarite...")
    fig_stationarity = plot_stationarity(log_returns, stationarity_results, output_dir=output_dir)
    results["fig_stationarity"] = fig_stationarity

    # 5. Serie temporelle
    print("\n[5/8] Generation du plot de serie temporelle...")
    fig_ts = plot_log_returns_time_series(df, log_returns, output_dir=output_dir)
    results["fig_time_series"] = fig_ts

    # 6. Autocorrelation des log-returns
    print("\n[6/8] Calcul de l'autocorrelation des log-returns...")
    fig_acf = compute_autocorrelation(log_returns, output_dir=output_dir)
    results["fig_acf"] = fig_acf

    # 7. Autocorrelation des log-returns² (volatilite clustering)
    print("\n[7/8] Calcul de l'autocorrelation des log-returns² (volatilite)...")
    fig_acf_sq = compute_autocorrelation_squared(log_returns, output_dir=output_dir)
    results["fig_acf_squared"] = fig_acf_sq

    # 8. Test de Ljung-Box (autocorrelation formelle)
    print("\n[8/8] Execution du test de Ljung-Box...")
    ljung_box_results = run_ljung_box_test(log_returns, output_dir=output_dir)
    results["ljung_box_test"] = ljung_box_results

    print("\n" + "=" * 80)
    print("ANALYSE TERMINEE")
    print("=" * 80)

    if show_plots:
        plt.show()

    return results


