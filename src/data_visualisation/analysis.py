"""Full analysis orchestration for dollar bars log-returns."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd

from src.constants import CLOSE_COLUMN, LOG_RETURN_COLUMN
from src.path import DOLLAR_BARS_PARQUET
from src.utils import get_logger

from .autocorrelation import (
    compute_autocorrelation,
    compute_autocorrelation_squared,
    run_ljung_box_test,
)
from .distribution import plot_log_returns_distribution, run_normality_tests
from .io import compute_log_returns, load_dollar_bars
from .stationarity import plot_stationarity, run_stationarity_tests
from .time_series import plot_log_returns_time_series

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

__all__ = [
    "run_full_analysis",
]


def run_full_analysis(
    parquet_path: Path = DOLLAR_BARS_PARQUET,
    output_dir: Path | None = None,
    show_plots: bool = True,
    sample_fraction: float = 1.0,
) -> dict[str, object]:
    """
    Execute complete log-returns analysis for dollar bars.

    Performs the following analysis steps:
    1. Distribution plot with histogram and Q-Q plot
    2. Normality tests (Jarque-Bera, Shapiro-Wilk)
    3. Stationarity tests (ADF, KPSS)
    4. Stationarity visualization with rolling statistics
    5. Time series plot with volatility
    6. Autocorrelation analysis (ACF/PACF)
    7. Squared returns autocorrelation (volatility clustering)
    8. Ljung-Box test for formal autocorrelation testing

    Args:
        parquet_path: Path to the parquet file containing dollar bars.
        output_dir: Directory to save results and plots.
        show_plots: If True, display plots interactively.
        sample_fraction: Fraction of dataset to use (0.0 to 1.0) for faster analysis.

    Returns:
        Dictionary containing all analysis results and figures.
    """
    logger.info("=" * 80)
    logger.info("ANALYSE DES LOG-RETURNS DES DOLLAR BARS")
    logger.info("=" * 80)

    # Load data
    df = load_dollar_bars(parquet_path)
    logger.info("Donnees chargees: %d barres", len(df))
    logger.info(
        "Periode: %s a %s",
        df["datetime_close"].min(),
        df["datetime_close"].max(),
    )

    # Apply systematic sampling if needed (preserves temporal structure)
    if sample_fraction < 1.0 and not df.empty:
        original_len = len(df)
        step = max(1, int(1 / sample_fraction))
        df = df.iloc[::step].reset_index(drop=True)
        logger.info(
            "Echantillonnage systematique applique: %d barres (1/%d = %.1f%% de l'original)",
            len(df),
            step,
            len(df) / original_len * 100,
        )
        if df.empty:
            raise ValueError("Dataset vide apres echantillonnage. Essayez d'augmenter sample_fraction.")

    # Load or compute log-returns
    logger.info("[0/8] Chargement des log-returns (naturels)...")
    if LOG_RETURN_COLUMN in df.columns:
        log_returns = pd.Series(df[LOG_RETURN_COLUMN])
        logger.info("Log-returns charges depuis la colonne existante (naturels)")
    else:
        log_returns = compute_log_returns(df, price_col=CLOSE_COLUMN)
        df[LOG_RETURN_COLUMN] = log_returns
        logger.info("Log-returns calcules (naturels) car colonne absente")

    n_valid = log_returns.dropna().shape[0]
    logger.info("Log-returns (naturels): %d observations", n_valid)
    logger.info("  Mean: %.6f", log_returns.mean())
    logger.info("  Std:  %.6f", log_returns.std())
    logger.info("  Min:  %.6f", log_returns.min())
    logger.info("  Max:  %.6f", log_returns.max())

    results: dict[str, object] = {
        "n_bars": len(df),
        "n_log_returns": n_valid,
        "log_returns": log_returns,
    }

    # 1. Distribution
    logger.info("[1/8] Generation du plot de distribution...")
    fig_dist = plot_log_returns_distribution(log_returns, output_dir=output_dir)
    results["fig_distribution"] = fig_dist

    # 2. Normality tests
    logger.info("[2/8] Execution des tests de normalite...")
    normality_results = run_normality_tests(log_returns, output_dir=output_dir)
    results["normality_tests"] = normality_results

    # 3. Stationarity tests
    logger.info("[3/8] Execution des tests de stationnarite...")
    stationarity_results = run_stationarity_tests(log_returns, output_dir=output_dir)
    results["stationarity_tests"] = stationarity_results

    # 4. Stationarity plot
    logger.info("[4/8] Generation du plot de stationnarite...")
    fig_stationarity = plot_stationarity(log_returns, stationarity_results, output_dir=output_dir)
    results["fig_stationarity"] = fig_stationarity

    # 5. Time series plot
    logger.info("[5/8] Generation du plot de serie temporelle...")
    fig_ts = plot_log_returns_time_series(df, log_returns, output_dir=output_dir)
    results["fig_time_series"] = fig_ts

    # 6. Autocorrelation of log-returns
    logger.info("[6/8] Calcul de l'autocorrelation des log-returns...")
    fig_acf = compute_autocorrelation(log_returns, output_dir=output_dir)
    results["fig_acf"] = fig_acf

    # 7. Autocorrelation of squared log-returns (volatility clustering)
    logger.info("[7/8] Calcul de l'autocorrelation des log-returns² (volatilite)...")
    fig_acf_sq = compute_autocorrelation_squared(log_returns, output_dir=output_dir)
    results["fig_acf_squared"] = fig_acf_sq

    # 8. Ljung-Box test
    logger.info("[8/8] Execution du test de Ljung-Box...")
    ljung_box_results = run_ljung_box_test(log_returns, output_dir=output_dir)
    results["ljung_box_test"] = ljung_box_results

    logger.info("=" * 80)
    logger.info("ANALYSE TERMINEE")
    logger.info("=" * 80)

    if show_plots:
        plt.show()

    return results
