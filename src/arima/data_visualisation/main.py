"""CLI entry point and orchestrator for ARIMA data visualizations.

This module serves as the main entry point for generating all ARIMA visualizations.
It imports and exposes functions from three specialized modules:
- pre_modeling: Visualizations before model fitting
- residual_diagnostics: Residual analysis and diagnostics
- model_performance: Model performance and prediction quality

Usage:
    python -m src.arima.data_visualisation.main
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd # type: ignore[import-untyped]


# Add project root to Python path for direct execution
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.arima.data_visualisation.data_loading import load_and_validate_data, load_series_for_year

# Import all visualization functions from submodules
from src.arima.data_visualisation.pre_modeling import (
    plot_acf_pacf,
    plot_log_returns_distribution,
    plot_stationarity,
    plot_stationarity_timeseries_with_bands,
    plot_weighted_series,
)
from src.arima.data_visualisation.plotting import (
    plot_seasonal_decomposition_daily,
    plot_seasonal_decomposition_monthly,
)
from src.arima.data_visualisation.residual_diagnostics import (
    plot_comprehensive_residuals,
    plot_residuals_acf,
    plot_residuals_histogram,
    plot_residuals_qq,
    plot_residuals_timeseries,
)
from src.constants import (
    ARIMA_DATA_VISU_PLOTS_DIR,
    ARIMA_EVALUATION_PLOTS_DIR,
    ARIMA_SEASONALITY_PLOTS_DIR,
    STATIONARITY_ROLLING_WINDOW_DEFAULT,
    WEIGHTED_LOG_RETURNS_FILE,
)
from src.utils import get_logger

logger = get_logger(__name__)

# Re-export all functions for convenient importing
__all__ = [
    # Pre-modeling visualizations
    "plot_weighted_series",
    "plot_log_returns_distribution",
    "plot_acf_pacf",
    "plot_stationarity",
    "plot_stationarity_timeseries_with_bands",
    # Returns visualizations
    # Residual diagnostics
    "plot_residuals_timeseries",
    "plot_residuals_histogram",
    "plot_residuals_qq",
    "plot_residuals_acf",
    "plot_comprehensive_residuals",
    # Model performance - imported locally to avoid circular imports
]


def _generate_seasonality_plots(data_file: str) -> None:
    """Generate daily and monthly seasonal decomposition plots.

    Args:
        data_file: Path to the weighted log returns CSV.

    Raises:
        ValueError: If the dataset does not contain any observations.
    """

    dataframe = load_and_validate_data(data_file, "log_return")
    if dataframe.empty:
        msg = "Dataset is empty; cannot generate seasonality plots"
        raise ValueError(msg)

    # Ensure index is datetime and get max date
    datetime_index = pd.DatetimeIndex(dataframe.index)
    max_date = datetime_index.max()
    if max_date is pd.NaT:
        msg = "No valid dates found in dataframe index"
        raise ValueError(msg)
    # max_date is a Timestamp, so we can access .year
    target_year = int(max_date.year)  # type: ignore[attr-defined]
    logger.info("Generating seasonal decomposition for %s", target_year)
    series = load_series_for_year(
        year=target_year,
        data_file=data_file,
        column="log_return",
    )

    ARIMA_SEASONALITY_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    daily_path = ARIMA_SEASONALITY_PLOTS_DIR / f"seasonality_daily_{target_year}.png"
    fig_daily, _ = plot_seasonal_decomposition_daily(series)
    fig_daily.savefig(daily_path, bbox_inches="tight")
    plt.close(fig_daily)
    logger.info("Daily seasonal decomposition saved to %s", daily_path)

    monthly_path = ARIMA_SEASONALITY_PLOTS_DIR / f"seasonality_monthly_{target_year}.png"
    fig_monthly, _ = plot_seasonal_decomposition_monthly(series)
    fig_monthly.savefig(monthly_path, bbox_inches="tight")
    plt.close(fig_monthly)
    logger.info("Monthly seasonal decomposition saved to %s", monthly_path)


def _execute_plot(
    plot_func: Callable[[], None],
    error_msg: str,
    raise_on_error: bool = False,
) -> None:
    """Execute a plot function with error handling.

    Args:
        plot_func: Function to execute.
        error_msg: Error message prefix.
        raise_on_error: If True, raise exception; otherwise log warning.
    """
    try:
        plot_func()
    except Exception as e:
        if raise_on_error:
            logger.error(f"{error_msg}: {e}", exc_info=True)
            raise
        logger.warning(f"{error_msg}: {e}", exc_info=True)


def generate_pre_modeling_plots() -> None:
    """Generate all pre-modeling visualizations.

    Creates plots for:
    - Weighted log-returns time series
    - Distribution of log-returns
    - ACF/PACF analysis
    - Stationarity analysis (4 separate publication-ready plots)
    - Returns clustering visualization (returns, |returns|, returns²)
    """
    logger.info("=" * 80)
    logger.info("GENERATING PRE-MODELING VISUALIZATIONS")
    logger.info("=" * 80)

    _execute_plot(
        lambda: plot_weighted_series(
            data_file=str(WEIGHTED_LOG_RETURNS_FILE),
            output_file=str(ARIMA_DATA_VISU_PLOTS_DIR / "weighted_log_returns_series.png"),
        ),
        "Failed to plot weighted series",
        raise_on_error=True,
    )

    _execute_plot(
        lambda: plot_log_returns_distribution(
            data_file=str(WEIGHTED_LOG_RETURNS_FILE),
            output_file=str(ARIMA_DATA_VISU_PLOTS_DIR / "log_returns_distribution.png"),
        ),
        "Failed to plot log returns distribution",
        raise_on_error=True,
    )

    _execute_plot(
        lambda: plot_acf_pacf(
            data_file=str(WEIGHTED_LOG_RETURNS_FILE),
            output_file=str(ARIMA_DATA_VISU_PLOTS_DIR / "acf_pacf.png"),
        ),
        "Failed to plot ACF/PACF",
        raise_on_error=True,
    )

    _execute_plot(
        lambda: plot_stationarity(
            data_file=str(WEIGHTED_LOG_RETURNS_FILE),
            output_dir=str(ARIMA_DATA_VISU_PLOTS_DIR),
            rolling_window=STATIONARITY_ROLLING_WINDOW_DEFAULT,
            alpha=0.05,
        ),
        "Failed to plot stationarity",
        raise_on_error=True,
    )

    _execute_plot(
        lambda: _generate_seasonality_plots(str(WEIGHTED_LOG_RETURNS_FILE)),
        "Failed to plot seasonal decomposition",
        raise_on_error=True,
    )

    logger.info("Pre-modeling visualizations completed")


def generate_residual_diagnostics(
    predictions_file: str | None = None,
    normality_tests_file: str | None = None,
    ljungbox_file: str | None = None,
    train_test_split_date: str | None = None,
) -> None:
    """Generate all residual diagnostic visualizations.

    Creates plots for:
    - Residuals time series
    - Residuals histogram with normality tests
    - Q-Q plot
    - ACF of residuals with Ljung-Box test
    - Comprehensive residuals dashboard (all above in one figure)

    Args:
        predictions_file: Path to rolling_predictions.csv. If None, uses default path.
        normality_tests_file: Path to normality_tests.json. If None, uses default path.
        ljungbox_file: Path to ljungbox_residuals.json. If None, uses default path.
        train_test_split_date: Optional date string for train/test split marker.
    """
    logger.info("=" * 80)
    logger.info("GENERATING RESIDUAL DIAGNOSTIC VISUALIZATIONS")
    logger.info("=" * 80)

    # Set default paths if not provided
    if predictions_file is None:
        predictions_file = str(
            _project_root / "results" / "arima" / "evaluation" / "rolling_predictions.csv"
        )

    if normality_tests_file is None:
        normality_tests_file = str(
            _project_root / "results" / "arima" / "evaluation" / "normality_tests.json"
        )

    if ljungbox_file is None:
        ljungbox_file = str(
            _project_root / "results" / "arima" / "evaluation" / "ljungbox_residuals.json"
        )

    # Check if predictions file exists
    if not Path(predictions_file).exists():
        logger.warning(
            f"Predictions file not found: {predictions_file}. "
            "Skipping residual diagnostics. Run ARIMA evaluation first."
        )
        return

    _execute_plot(
        lambda: plot_residuals_timeseries(
            predictions_file=predictions_file,
            output_file=str(ARIMA_EVALUATION_PLOTS_DIR / "residuals_timeseries.png"),
            train_test_split_date=train_test_split_date,
        ),
        "Failed to plot residuals time series",
    )

    _execute_plot(
        lambda: plot_residuals_histogram(
            predictions_file=predictions_file,
            output_file=str(ARIMA_EVALUATION_PLOTS_DIR / "residuals_histogram.png"),
            normality_tests_file=normality_tests_file,
        ),
        "Failed to plot residuals histogram",
    )

    _execute_plot(
        lambda: plot_residuals_qq(
            predictions_file=predictions_file,
            output_file=str(ARIMA_EVALUATION_PLOTS_DIR / "residuals_qq.png"),
        ),
        "Failed to plot residuals Q-Q",
    )

    _execute_plot(
        lambda: plot_residuals_acf(
            predictions_file=predictions_file,
            output_file=str(ARIMA_EVALUATION_PLOTS_DIR / "residuals_acf.png"),
            ljungbox_file=ljungbox_file,
        ),
        "Failed to plot residuals ACF",
    )

    _execute_plot(
        lambda: plot_comprehensive_residuals(
            predictions_file=predictions_file,
            output_file=str(ARIMA_EVALUATION_PLOTS_DIR / "residuals_comprehensive.png"),
            normality_tests_file=normality_tests_file,
            ljungbox_file=ljungbox_file,
            train_test_split_date=train_test_split_date,
        ),
        "Failed to plot comprehensive residuals dashboard",
    )

    logger.info("Residual diagnostic visualizations completed")


def generate_performance_plots(
    predictions_file: str | None = None,
    metrics_file: str | None = None,
    train_test_split_date: str | None = None,
) -> None:
    """Generate all model performance visualizations.

    Creates plots for:
    - Fitted vs actual values
    - Predictions vs actual values (with train/test split)
    - Forecast errors analysis

    Args:
        predictions_file: Path to rolling_predictions.csv. If None, uses default path.
        metrics_file: Path to rolling_metrics.json. If None, uses default path.
        train_test_split_date: Date string for train/test split. Required for some plots.
    """
    logger.info("=" * 80)
    logger.info("GENERATING MODEL PERFORMANCE VISUALIZATIONS")
    logger.info("=" * 80)

    # Set default paths if not provided
    if predictions_file is None:
        predictions_file = str(
            _project_root / "results" / "arima" / "evaluation" / "rolling_predictions.csv"
        )

    if metrics_file is None:
        metrics_file = str(
            _project_root / "results" / "arima" / "evaluation" / "rolling_metrics.json"
        )

    # Check if predictions file exists
    if not Path(predictions_file).exists():
        logger.warning(
            f"Predictions file not found: {predictions_file}. "
            "Skipping performance plots. Run ARIMA evaluation first."
        )
        return

    _execute_plot(
        lambda: _import_and_plot_fitted_vs_actual(
            predictions_file=predictions_file,
            output_file=str(ARIMA_EVALUATION_PLOTS_DIR / "fitted_vs_actual.png"),
        ),
        "Failed to plot fitted vs actual",
    )

    if train_test_split_date is not None:
        _execute_plot(
            lambda: _import_and_plot_predictions_vs_actual(
                predictions_file=predictions_file,
                output_file=str(ARIMA_EVALUATION_PLOTS_DIR / "predictions_vs_actual.png"),
                train_test_split_date=train_test_split_date,
                metrics_file=metrics_file,
            ),
            "Failed to plot predictions vs actual",
        )
    else:
        logger.warning("train_test_split_date not provided. Skipping predictions vs actual plot.")

    _execute_plot(
        lambda: _import_and_plot_forecast_errors(
            predictions_file=predictions_file,
            output_file=str(ARIMA_EVALUATION_PLOTS_DIR / "forecast_errors.png"),
            train_test_split_date=train_test_split_date,
        ),
        "Failed to plot forecast errors",
    )

    logger.info("Model performance visualizations completed")


def generate_all_plots(
    predictions_file: str | None = None,
    normality_tests_file: str | None = None,
    ljungbox_file: str | None = None,
    metrics_file: str | None = None,
    train_test_split_date: str | None = None,
) -> None:
    """Generate ALL ARIMA visualizations in one go.

    This convenience function generates:
    1. All pre-modeling plots
    2. All residual diagnostic plots (if evaluation data available)
    3. All model performance plots (if evaluation data available)

    Args:
        predictions_file: Path to rolling_predictions.csv. If None, uses default.
        normality_tests_file: Path to normality_tests.json. If None, uses default.
        ljungbox_file: Path to ljungbox_residuals.json. If None, uses default.
        metrics_file: Path to rolling_metrics.json. If None, uses default.
        train_test_split_date: Date string for train/test split visualization.
    """
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING ALL ARIMA VISUALIZATIONS")
    logger.info("=" * 80 + "\n")

    # Always generate pre-modeling plots (they don't depend on model results)
    generate_pre_modeling_plots()

    # Generate post-modeling plots if data is available
    generate_residual_diagnostics(
        predictions_file=predictions_file,
        normality_tests_file=normality_tests_file,
        ljungbox_file=ljungbox_file,
        train_test_split_date=train_test_split_date,
    )

    generate_performance_plots(
        predictions_file=predictions_file,
        metrics_file=metrics_file,
        train_test_split_date=train_test_split_date,
    )

    logger.info("\n" + "=" * 80)
    logger.info("ALL ARIMA VISUALIZATIONS COMPLETED")
    logger.info("=" * 80 + "\n")


def main() -> None:
    """Main CLI function to generate pre-modeling visualizations.

    This is the default behavior when running the module directly.
    For complete visualization generation (including residuals and performance),
    use generate_all_plots() after model evaluation.
    """
    logger.info("Running ARIMA data visualization (pre-modeling only)")
    logger.info("For complete visualizations including residuals and performance,")
    logger.info("run generate_all_plots() after model evaluation")

    generate_pre_modeling_plots()


def _import_and_plot_fitted_vs_actual(
    predictions_file: str,
    output_file: str,
) -> None:
    """Import and plot fitted vs actual values (to avoid circular import)."""
    from src.arima.evaluation_arima.model_performance import plot_fitted_vs_actual

    plot_fitted_vs_actual(predictions_file=predictions_file, output_file=output_file)


def _import_and_plot_predictions_vs_actual(
    predictions_file: str,
    output_file: str,
    train_test_split_date: str,
    metrics_file: str | None = None,
) -> None:
    """Import and plot predictions vs actual values (to avoid circular import)."""
    from src.arima.evaluation_arima.model_performance import plot_predictions_vs_actual

    plot_predictions_vs_actual(
        predictions_file=predictions_file,
        output_file=output_file,
        train_test_split_date=train_test_split_date,
        metrics_file=metrics_file,
    )


def _import_and_plot_forecast_errors(
    predictions_file: str,
    output_file: str,
    train_test_split_date: str | None = None,
) -> None:
    """Import and plot forecast errors (to avoid circular import)."""
    from src.arima.evaluation_arima.model_performance import plot_forecast_errors

    plot_forecast_errors(
        predictions_file=predictions_file,
        output_file=output_file,
        train_test_split_date=train_test_split_date,
    )


if __name__ == "__main__":
    main()
