"""Model performance visualizations for ARIMA models.

This module contains visualizations for evaluating ARIMA model performance:
- Fitted vs actual values (training set)
- Predictions vs actual values (test set)
- Forecast errors analysis
"""

from __future__ import annotations

from matplotlib.axes import Axes
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.constants import (  # type: ignore
    COLOR_ACTUAL,
    COLOR_PREDICTION,
    COLOR_RESIDUAL,
    COLOR_SPLIT_LINE,
    COLOR_TEST,
    COLOR_TRAIN,
    FONTSIZE_AXIS,
    FONTSIZE_LABEL,
    FONTSIZE_SUBTITLE,
    FONTSIZE_TITLE,
    LINEWIDTH_DEFAULT,
    MAX_POINTS_SUBSAMPLE,
    PLOT_ALPHA_DEFAULT,
    PLOT_ALPHA_LIGHT,
    STATISTICS_PRECISION,
    TEXTBOX_STYLE_INFO,
)
from src.utils import get_logger
from src.visualization import (
    add_grid,
    add_metrics_textbox,
    add_zero_line,
    create_standard_figure,
    load_json_if_exists,
    subsample_for_plotting,
)

from src.arima.data_visualisation.data_loading import (
    load_predictions_with_columns,
    load_predictions_with_split,
)
from src.arima.data_visualisation.plotting import _save_plot

logger = get_logger(__name__)


def _calculate_fit_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Calculate fit metrics for model evaluation."""
    residuals = y_true - y_pred
    mse = float((residuals**2).mean())
    rmse = float(np.sqrt(mse))
    mae = float(residuals.abs().mean())
    r2 = float(1 - (residuals**2).sum() / ((y_true - y_true.mean()) ** 2).sum())

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R²": r2}


def plot_fitted_vs_actual(
    predictions_file: str,
    output_file: str,
    max_points: int | None = MAX_POINTS_SUBSAMPLE,
) -> None:
    """Plot fitted values vs actual values from ARIMA model.

    This visualization shows how well the model fits the training data.

    Args:
        predictions_file: Path to rolling_predictions.csv with y_true and y_pred columns.
        output_file: Path to save the plot.
        max_points: Maximum number of points to plot for readability. If None, plot all.

    Raises:
        FileNotFoundError: If predictions_file does not exist.
        ValueError: If required columns are missing.
    """
    logger.info("Loading predictions data for fitted vs actual plot")

    df: pd.DataFrame = load_predictions_with_columns(
        predictions_file,
        required_columns=["y_true", "y_pred"],
        index_col="date",
        parse_dates=True,
    )

    # Subsample if needed
    if max_points is not None and len(df) > max_points:
        logger.info(f"Subsampling {len(df)} points to {max_points} for better visualization")
        df = subsample_for_plotting(df, max_points=max_points, method="uniform")

    # Create figure
    _, ax = create_standard_figure(figsize=(14, 6))

    # Plot
    ax.plot(
        df.index,
        df["y_true"],
        color=COLOR_ACTUAL,
        linewidth=LINEWIDTH_DEFAULT,
        alpha=PLOT_ALPHA_DEFAULT,
        label="Actual values",
    )

    ax.plot(
        df.index,
        df["y_pred"],
        color=COLOR_TEST,
        linewidth=LINEWIDTH_DEFAULT,
        alpha=0.7,
        label="Fitted values",
        linestyle="--",
    )

    ax.fill_between(
        df.index,
        df["y_true"],
        df["y_pred"],
        alpha=0.2,
        color="gray",
        label="Difference",
    )

    add_zero_line(ax)

    # Calculate and display metrics
    metrics = _calculate_fit_metrics(df.loc[:, "y_true"], df.loc[:, "y_pred"])
    add_metrics_textbox(
        ax,
        metrics,
        position=(0.02, 0.98),
        precision=STATISTICS_PRECISION,
        style=TEXTBOX_STYLE_INFO,
    )

    ax.set_title(
        "ARIMA: Valeurs ajustées vs Valeurs réelles",
        fontsize=FONTSIZE_TITLE,
        fontweight="bold",
    )
    ax.set_xlabel("Date", fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Log-return", fontsize=FONTSIZE_LABEL)
    ax.legend(loc="upper right", fontsize=10)
    add_grid(ax, alpha=PLOT_ALPHA_LIGHT, linestyle="--")

    plt.tight_layout()
    _save_plot(output_file)
    logger.info(
        "Fitted vs actual plot saved: RMSE=%.8f, MAE=%.8f, R²=%.4f",
        metrics["RMSE"],
        metrics["MAE"],
        metrics["R²"],
    )


def _plot_train_test_predictions(
    ax: Axes,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_date: pd.Timestamp,
) -> None:
    """Plot train and test predictions on given axes."""
    # Training period
    if not train_df.empty:
        ax.plot(
            train_df.index,
            train_df["y_true"],
            color=COLOR_TRAIN,
            linewidth=LINEWIDTH_DEFAULT,
            alpha=PLOT_ALPHA_DEFAULT,
            label="Actual (train)",
        )
        ax.plot(
            train_df.index,
            train_df["y_pred"],
            color=COLOR_TRAIN,
            linewidth=LINEWIDTH_DEFAULT,
            alpha=0.4,
            linestyle="--",
            label="Fitted (train)",
        )

    # Test period
    if not test_df.empty:
        ax.plot(
            test_df.index,
            test_df["y_true"],
            color=COLOR_TEST,
            linewidth=LINEWIDTH_DEFAULT + 0.5,
            alpha=PLOT_ALPHA_DEFAULT,
            label="Actual (test)",
        )
        ax.plot(
            test_df.index,
            test_df["y_pred"],
            color=COLOR_PREDICTION,
            linewidth=LINEWIDTH_DEFAULT,
            alpha=0.8,
            linestyle="-",
            label="Predictions (test)",
        )

    # Split line
    ax.axvline(
        float(mdates.date2num(split_date.to_pydatetime())),
        color=COLOR_SPLIT_LINE,
        linestyle="--",
        linewidth=2.5,
        alpha=0.7,
        label=f"Train/Test split: {split_date.date()}",
    )


def plot_predictions_vs_actual(
    predictions_file: str,
    output_file: str,
    train_test_split_date: str,
    metrics_file: str | None = None,
) -> None:
    """Plot predictions vs actual values with train/test split visualization.

    Args:
        predictions_file: Path to rolling_predictions.csv with y_true and y_pred columns.
        output_file: Path to save the plot.
        train_test_split_date: Date string marking the train/test boundary.
        metrics_file: Optional path to rolling_metrics.json for evaluation metrics.

    Raises:
        FileNotFoundError: If predictions_file does not exist.
        ValueError: If required columns are missing or split date invalid.
    """
    logger.info("Loading predictions data for predictions vs actual plot")

    train_df, test_df = load_predictions_with_split(
        predictions_file,
        train_test_split_date,
        value_columns=["y_true", "y_pred"],
    )

    split_date = pd.to_datetime(train_test_split_date)

    # Create figure
    _, ax = create_standard_figure(figsize=(14, 6))

    _plot_train_test_predictions(ax, train_df, test_df, split_date)
    add_zero_line(ax)

    # Load and display metrics
    if metrics_file is not None and not test_df.empty:
        metrics_data = load_json_if_exists(metrics_file)
        if metrics_data:
            test_metrics = (
                metrics_data.get("test", metrics_data) if "test" in metrics_data else metrics_data
            )

            display_metrics = {
                "RMSE": test_metrics.get("rmse", 0),
                "MAE": test_metrics.get("mae", 0),
                "MSE": test_metrics.get("mse", 0),
            }

            add_metrics_textbox(
                ax,
                display_metrics,
                position=(0.98, 0.98),
                precision=STATISTICS_PRECISION,
                style=TEXTBOX_STYLE_INFO,
                ha="right",
            )

    ax.set_title(
        "ARIMA: Prédictions vs Valeurs réelles (Train/Test)",
        fontsize=FONTSIZE_TITLE,
        fontweight="bold",
    )
    ax.set_xlabel("Date", fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Log-return", fontsize=FONTSIZE_LABEL)
    ax.legend(loc="best", fontsize=9, ncol=2)
    add_grid(ax, alpha=PLOT_ALPHA_LIGHT, linestyle="--")

    plt.tight_layout()
    _save_plot(output_file)
    logger.info(
        "Predictions vs actual plot saved with train/test split at %s",
        split_date.date(),
    )


def _plot_forecast_errors_timeseries(
    ax: Axes,
    residuals: pd.Series,
    split_date: pd.Timestamp | None,
) -> tuple[float, float]:
    """Plot forecast errors time series on given axes."""
    if split_date is not None:
        train_errors = residuals.loc[residuals.index < split_date]
        test_errors = residuals.loc[residuals.index >= split_date]

        if not train_errors.empty:
            ax.plot(
                train_errors.index,
                train_errors,
                color=COLOR_TRAIN,
                linewidth=1,
                alpha=0.8,
                label="Train errors",
            )
        if not test_errors.empty:
            ax.plot(
                test_errors.index,
                test_errors,
                color=COLOR_TEST,
                linewidth=1,
                alpha=0.8,
                label="Test errors",
            )
        ax.axvline(
            float(mdates.date2num(split_date.to_pydatetime())),
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
        )
    else:
        ax.plot(
            residuals.index,
            residuals,
            color=COLOR_RESIDUAL,
            linewidth=1,
            alpha=0.8,
        )

    ax.axhline(0, color="black", linestyle="-", linewidth=1, alpha=0.6)

    mean_err = float(residuals.mean())
    std_err = float(residuals.std())

    return mean_err, std_err


def _plot_forecast_errors_distribution(
    ax: Axes,
    residuals: pd.Series,
    mean_err: float,
    std_err: float,
) -> None:
    """Plot forecast errors distribution on given axes."""
    ax.hist(
        residuals,
        bins=40,
        density=False,
        alpha=0.7,
        color=COLOR_RESIDUAL,
        edgecolor="black",
        linewidth=0.5,
    )

    # Add mean and std lines
    ax.axvline(
        mean_err,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_err:.6f}",
    )
    ax.axvline(
        mean_err + std_err,
        color="orange",
        linestyle=":",
        linewidth=1.5,
        label=f"±1σ: {std_err:.6f}",
    )
    ax.axvline(mean_err - std_err, color="orange", linestyle=":", linewidth=1.5)

    # Add statistics
    p25 = float(np.percentile(residuals, 25))
    p50 = float(np.percentile(residuals, 50))
    p75 = float(np.percentile(residuals, 75))

    stats_text = (
        f"Statistics:\n"
        f"Mean: {mean_err:.6f}\n"
        f"Std: {std_err:.6f}\n"
        f"25%: {p25:.6f}\n"
        f"50%: {p50:.6f}\n"
        f"75%: {p75:.6f}"
    )

    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        ha="right",
        bbox=TEXTBOX_STYLE_INFO,
    )


def plot_forecast_errors(
    predictions_file: str,
    output_file: str,
    train_test_split_date: str | None = None,
) -> None:
    """Plot forecast errors (residuals) over time with distribution analysis.

    Creates a 2-panel visualization:
    - Left: Forecast errors time series
    - Right: Distribution of forecast errors

    Args:
        predictions_file: Path to rolling_predictions.csv with residual column.
        output_file: Path to save the plot.
        train_test_split_date: Optional date string to mark train/test boundary.

    Raises:
        FileNotFoundError: If predictions_file does not exist.
        ValueError: If required columns are missing.
    """
    logger.info("Loading predictions data for forecast errors analysis")

    df = load_predictions_with_columns(
        predictions_file,
        required_columns=["residual"],
        index_col="date",
        parse_dates=True,
    )

    residuals = df.loc[:, "residual"]

    split_date = pd.to_datetime(train_test_split_date) if train_test_split_date else None

    # Create 2-panel figure
    fig, axes = create_standard_figure(n_rows=1, n_cols=2, figsize=(14, 5))

    # Left: Time series
    mean_err, std_err = _plot_forecast_errors_timeseries(axes[0], residuals, split_date)

    axes[0].set_title(
        f"Erreurs de prévision (μ={mean_err:.6f}, σ={std_err:.6f})",
        fontsize=FONTSIZE_SUBTITLE,
        fontweight="bold",
    )
    axes[0].set_xlabel("Date", fontsize=FONTSIZE_AXIS)
    axes[0].set_ylabel("Forecast Error", fontsize=FONTSIZE_AXIS)
    if train_test_split_date:
        axes[0].legend(fontsize=9)
    axes[0].grid(alpha=PLOT_ALPHA_LIGHT, linestyle="--")

    # Right: Distribution
    _plot_forecast_errors_distribution(axes[1], residuals, mean_err, std_err)

    axes[1].set_title(
        "Distribution des erreurs",
        fontsize=FONTSIZE_SUBTITLE,
        fontweight="bold",
    )
    axes[1].set_xlabel("Forecast Error", fontsize=FONTSIZE_AXIS)
    axes[1].set_ylabel("Frequency", fontsize=FONTSIZE_AXIS)
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=PLOT_ALPHA_LIGHT, linestyle="--")

    # Overall title
    fig.suptitle(
        "Analyse des erreurs de prévision ARIMA",
        fontsize=FONTSIZE_TITLE,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    _save_plot(output_file)
    logger.info(
        "Forecast errors plot saved: mean=%.6f, std=%.6f",
        mean_err,
        std_err,
    )
