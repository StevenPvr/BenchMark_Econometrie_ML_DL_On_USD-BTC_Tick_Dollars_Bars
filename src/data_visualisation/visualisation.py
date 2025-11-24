"""Visualisation module for dollar bars log-returns analysis."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from statsmodels.tsa.stattools import adfuller, kpss  # type: ignore
from scipy import stats  # type: ignore

from ..constants import CLOSE_COLUMN, LOG_RETURN_COLUMN
from ..path import DOLLAR_IMBALANCE_BARS_PARQUET, LOG_RETURNS_PARQUET, LOG_RETURNS_CSV


def load_dollar_bars(parquet_path: Path = DOLLAR_IMBALANCE_BARS_PARQUET) -> pd.DataFrame:
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
    ax1.hist(data, bins=50, alpha=0.7, color="steelblue", edgecolor="white", density=True)

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


def run_full_analysis(
    parquet_path: Path = DOLLAR_IMBALANCE_BARS_PARQUET,
    output_dir: Optional[Path] = None,
    show_plots: bool = True,
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

    # Calculer les log-returns
    print("\n[0/4] Calcul des log-returns...")
    log_returns = compute_log_returns(df, price_col=CLOSE_COLUMN)
    df[LOG_RETURN_COLUMN] = log_returns

    n_valid = log_returns.dropna().shape[0]
    print(f"Log-returns calcules: {n_valid} observations")
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
    print("\n[1/5] Generation du plot de distribution...")
    fig_dist = plot_log_returns_distribution(log_returns, output_dir=output_dir)
    results["fig_distribution"] = fig_dist

    # 2. Tests de stationnarite
    print("\n[2/5] Execution des tests de stationnarite...")
    stationarity_results = run_stationarity_tests(log_returns, output_dir=output_dir)
    results["stationarity_tests"] = stationarity_results

    # 3. Plot de stationnarite
    print("\n[3/5] Generation du plot de stationnarite...")
    fig_stationarity = plot_stationarity(log_returns, stationarity_results, output_dir=output_dir)
    results["fig_stationarity"] = fig_stationarity

    # 4. Serie temporelle
    print("\n[4/5] Generation du plot de serie temporelle...")
    fig_ts = plot_log_returns_time_series(df, log_returns, output_dir=output_dir)
    results["fig_time_series"] = fig_ts

    # 5. Autocorrelation
    print("\n[5/5] Calcul de l'autocorrelation...")
    fig_acf = compute_autocorrelation(log_returns, output_dir=output_dir)
    results["fig_acf"] = fig_acf

    # Sauvegarder le dataset avec log-returns dans data/prepared/
    log_returns_df = df.copy()
    log_returns_df[LOG_RETURN_COLUMN] = log_returns

    # Sauvegarder en parquet (complet) et CSV (sample)
    LOG_RETURNS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    log_returns_df.to_parquet(LOG_RETURNS_PARQUET, index=False)
    print(f"\nDataset avec log-returns sauvegarde: {LOG_RETURNS_PARQUET}")

    # CSV sample (10% pour inspection)
    sample_size = max(1, int(len(log_returns_df) * 0.1))
    log_returns_df.head(sample_size).to_csv(LOG_RETURNS_CSV, index=False)
    print(f"CSV sample (10%) sauvegarde: {LOG_RETURNS_CSV}")

    print("\n" + "=" * 80)
    print("ANALYSE TERMINEE")
    print("=" * 80)

    if show_plots:
        plt.show()

    return results


# =============================================================================
# GARCH Visualization Utilities
# =============================================================================


def create_figure_canvas(
    figsize: tuple[int, int] = (10, 6),
    n_rows: int = 1,
    n_cols: int = 1,
) -> tuple[Figure, Figure, np.ndarray]:
    """Create matplotlib figure and canvas with subplots.

    Args:
        figsize: Figure size (width, height).
        n_rows: Number of rows in subplot grid.
        n_cols: Number of columns in subplot grid.

    Returns:
        Tuple of (figure, canvas_figure, axes_array).
    """
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    # For backward compatibility, return fig as both figure and canvas
    return fig, fig, axes


def save_canvas(
    canvas: Figure,
    path: Path | str,
    format: str = "png",
    dpi: int = 150,
) -> None:
    """Save matplotlib canvas to file.

    Args:
        canvas: Matplotlib Figure object.
        path: Output path.
        format: Image format.
        dpi: Resolution.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.savefig(path, format=format, dpi=dpi, bbox_inches="tight")


def add_zero_line(
    ax: Axes,
    color: str = "black",
    linewidth: float = 1.0,
    linestyle: str = "-",
) -> None:
    """Add a horizontal zero line to the plot.

    Args:
        ax: Matplotlib axes object.
        color: Line color.
        linewidth: Line width.
        linestyle: Line style.
    """
    ax.axhline(y=0, color=color, linewidth=linewidth, linestyle=linestyle)


def prepare_temporal_axis(
    dates: Sequence[Any] | np.ndarray | pd.Series | pd.Index | None,
    length: int | None = None,
) -> np.ndarray:
    """Prepare temporal axis for plotting.

    Args:
        dates: Date values.
        length: Desired length (if None, use dates length).

    Returns:
        Array for x-axis.
    """
    if length is None:
        if dates is not None and hasattr(dates, "__len__"):
            length = len(dates)
        else:
            length = 100  # default

    return np.arange(length)


def plot_histogram_with_normal_overlay(
    residuals: np.ndarray,
    ax: Axes | None = None,
    bins: int = 50,
    alpha: float = 0.7,
) -> tuple[float, float]:
    """Plot histogram of residuals with normal distribution overlay.

    Args:
        residuals: Residual values.
        ax: Matplotlib axes (if None, use current axes).
        bins: Number of histogram bins.
        alpha: Transparency level.

    Returns:
        Tuple of (mean, std) of residuals.
    """
    if ax is None:
        ax = plt.gca()

    # Plot histogram
    ax.hist(residuals, bins=bins, alpha=alpha, density=True,
            color="skyblue", edgecolor="black", linewidth=0.5)

    # Add normal distribution overlay
    mean_val = float(np.mean(residuals))
    std_val = float(np.std(residuals))

    x = np.linspace(mean_val - 3*std_val, mean_val + 3*std_val, 100)
    y = 1/(std_val * np.sqrt(2*np.pi)) * np.exp(-0.5*((x-mean_val)/std_val)**2)

    ax.plot(x, y, color="red", linewidth=2, label="Distribution normale")
    ax.legend()

    return mean_val, std_val
