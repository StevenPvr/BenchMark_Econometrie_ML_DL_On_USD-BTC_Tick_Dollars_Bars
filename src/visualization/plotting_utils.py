"""Common plotting utilities for ARIMA and GARCH visualizations.

This module provides reusable plotting functions to maintain DRY principles
across the codebase. All visualization modules should use these utilities
instead of reimplementing common patterns.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence, cast
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import get_logger, save_plot

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

logger = get_logger(__name__)
_DATE_AXIS_ROTATION = 45
_DATE_AXIS_YEAR_LOCATOR_MONTHS = (1, 7)
_DATE_AXIS_DAILY_MONTHS = (1, 4, 7, 10)
_PLOT_DPI = 300
_PLOT_LINEWIDTH_THIN = 0.8
_TEXTBOX_STYLE_DEFAULT = {"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8}
_TEXTBOX_STYLE_INFO = {"boxstyle": "round", "facecolor": "lightblue", "alpha": 0.8}
_TEXTBOX_STYLE_SUCCESS = {"boxstyle": "round", "facecolor": "lightgreen", "alpha": 0.7}
_TEXTBOX_STYLE_ERROR = {"boxstyle": "round", "facecolor": "lightcoral", "alpha": 0.7}


# ============================================================================
# Figure and Canvas Creation
# ============================================================================


def create_figure_canvas(
    figsize: tuple[float, float],
    n_rows: int = 1,
    n_cols: int = 1,
    *,
    constrained_layout: bool = True,
    backend: str = "agg",
) -> tuple[Figure, FigureCanvasAgg, Any]:
    """Create matplotlib figure and canvas with subplots.

    Uses non-interactive backend (Agg) for server/batch environments.

    Args:
        figsize: Figure size as (width, height) in inches.
        n_rows: Number of subplot rows.
        n_cols: Number of subplot columns.
        constrained_layout: Use constrained layout for better spacing.
        backend: Backend to use ('agg' or 'default').

    Returns:
        Tuple of (figure, canvas, axes). axes is a single Axes if n_rows=n_cols=1,
        otherwise an array of Axes.

    Raises:
        ImportError: If matplotlib is unavailable.
    """
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from matplotlib.figure import Figure

        fig = Figure(figsize=figsize, constrained_layout=constrained_layout)
        canvas = FigureCanvas(fig)

        if n_rows == 1 and n_cols == 1:
            axes = fig.add_subplot(1, 1, 1)
        else:
            axes = fig.subplots(n_rows, n_cols)

        return fig, canvas, axes
    except Exception as ex:
        msg = f"Matplotlib unavailable: {ex}"
        raise ImportError(msg) from ex


def create_standard_figure(
    n_rows: int = 1,
    n_cols: int = 1,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, Any]:
    """Create standard matplotlib figure with pyplot backend.

    Use this for interactive plotting or when using plt.show().

    Args:
        n_rows: Number of subplot rows.
        n_cols: Number of subplot columns.
        figsize: Figure size. If None, uses matplotlib defaults.

    Returns:
        Tuple of (figure, axes).
    """
    if figsize is not None:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    else:
        fig, axes = plt.subplots(n_rows, n_cols)
    return fig, axes


# ============================================================================
# Figure Saving
# ============================================================================


def save_figure(
    fig: Figure | None = None,
    output_path: str | Path | None = None,
    *,
    dpi: int = _PLOT_DPI,
    bbox_inches: str = "tight",
    close_after: bool = True,
    **kwargs: Any,
) -> None:
    """Save matplotlib figure to file with consistent settings.

    Args:
        fig: Figure to save. If None, uses plt.gcf().
        output_path: Output file path.
        dpi: Dots per inch for rasterized output.
        bbox_inches: Bounding box setting ('tight' or None).
        close_after: Close figure after saving.
        **kwargs: Additional arguments passed to savefig.
    """
    if output_path is None:
        msg = "output_path is required"
        raise ValueError(msg)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fig is None:
        fig = plt.gcf()

    fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches, **kwargs)

    if close_after:
        plt.close(fig)

    logger.info(f"Saved plot to: {output_path}")


def save_canvas(
    canvas: FigureCanvasAgg,
    output_path: str | Path,
    format: str = "png",
) -> None:
    """Save matplotlib canvas to file.

    Args:
        canvas: Matplotlib canvas to save.
        output_path: Output file path.
        format: Output format ('png', 'pdf', 'svg', etc.).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    supported_formats = {"png", "pdf", "svg"}

    if format not in supported_formats:
        msg = f"Unsupported format: {format}"
        raise ValueError(msg)

    canvas.figure.savefig(str(output_path), format=format)

    logger.info(f"Saved canvas to: {output_path}")


def save_plot_wrapper(
    output_file: str | Path,
    *,
    dpi: int = _PLOT_DPI,
    **kwargs: Any,
) -> None:
    """Wrapper for src.utils.save_plot() with consistent defaults.

    Args:
        output_file: Path to save the plot.
        dpi: Dots per inch.
        **kwargs: Additional arguments passed to save_plot.
    """
    save_plot(output_file, dpi=dpi, close_after=True, **kwargs)


# ============================================================================
# Date Axis Formatting
# ============================================================================


def format_date_axis(
    ax: Axes,
    *,
    major_locator: mdates.DateLocator | None = None,
    major_formatter: mdates.DateFormatter | None = None,
    minor_locator: mdates.DateLocator | None = None,
    rotation: float = _DATE_AXIS_ROTATION,
    ha: str = "right",
) -> None:
    """Format x-axis for dates with standard settings.

    Args:
        ax: Matplotlib axes to format.
        major_locator: Major tick locator. If None, uses YearLocator.
        major_formatter: Major tick formatter. If None, uses DateFormatter("%Y").
        minor_locator: Minor tick locator. If None, uses MonthLocator.
        rotation: Label rotation angle in degrees.
        ha: Horizontal alignment ('left', 'center', 'right').
    """
    if major_locator is None:
        major_locator = mdates.YearLocator()
    if major_formatter is None:
        major_formatter = mdates.DateFormatter("%Y")
    if minor_locator is None:
        minor_locator = mdates.MonthLocator(_DATE_AXIS_YEAR_LOCATOR_MONTHS)

    ax.xaxis.set_major_locator(major_locator)
    ax.xaxis.set_major_formatter(major_formatter)
    ax.xaxis.set_minor_locator(minor_locator)
    for label in ax.xaxis.get_majorticklabels():
        label.set_rotation(rotation)
        label.set_horizontalalignment(cast(Literal["left", "center", "right"], ha))


def format_seasonal_axis_daily(ax: Axes, rotation: float = _DATE_AXIS_ROTATION) -> None:
    """Format x-axis for daily seasonal plots (monthly ticks).

    Args:
        ax: Matplotlib axes to format.
        rotation: Label rotation angle in degrees.
    """
    ax.xaxis.set_major_locator(mdates.MonthLocator(_DATE_AXIS_DAILY_MONTHS))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    for label in ax.xaxis.get_majorticklabels():
        label.set_rotation(rotation)
        label.set_horizontalalignment("right")


# ============================================================================
# Data Preparation and Validation
# ============================================================================


def prepare_temporal_axis(
    dates: Sequence[Any] | np.ndarray | pd.Series | pd.Index | None,
    length: int,
) -> np.ndarray:
    """Prepare x-axis values from dates or create numeric index.

    Args:
        dates: Optional sequence, array, series, or index of dates.
        length: Expected length of data.

    Returns:
        X-axis values array (dates or numeric indices).

    Raises:
        ValueError: If date conversion fails critically.
    """
    if dates is None:
        return np.arange(length, dtype=float)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            dates_clean = pd.to_datetime(dates, errors="coerce")

        # Cast to ensure type checker understands this can have .values attribute
        dates_clean_with_values = cast(Any, dates_clean)
        dates_values = (
            dates_clean_with_values.values
            if hasattr(dates_clean_with_values, "values")
            else dates_clean_with_values
        )

        # Check if all dates are invalid (all NaT)
        if np.all(pd.isna(dates_values)):
            logger.warning("All dates are invalid (NaT), using numeric indices")
            return np.arange(length, dtype=float)

        return np.asarray(dates_values, dtype=object)
    except Exception as e:
        logger.warning(f"Failed to convert dates: {e}. Using numeric indices.")
        return np.arange(length, dtype=float)


def _check_arrays_non_empty(arrays: tuple[np.ndarray, ...]) -> bool:
    """Check that all arrays are non-empty."""
    for i, arr in enumerate(arrays):
        if arr.size == 0:
            logger.warning(f"Array {i} is empty")
            return False
    return True


def _check_arrays_same_length(arrays: tuple[np.ndarray, ...]) -> bool:
    """Check that all arrays have the same length."""
    if len(arrays) <= 1:
        return True
    lengths = [arr.size for arr in arrays]
    if len(set(lengths)) > 1:
        logger.warning(f"Arrays have different lengths: {lengths}")
        return False
    return True


def _check_arrays_finite(arrays: tuple[np.ndarray, ...]) -> bool:
    """Check that all array values are finite."""
    for i, arr in enumerate(arrays):
        if not np.all(np.isfinite(arr)):
            n_invalid = np.sum(~np.isfinite(arr))
            logger.warning(f"Array {i} has {n_invalid} non-finite values")
            return False
    return True


def validate_plot_arrays(
    *arrays: np.ndarray,
    check_finite: bool = True,
    check_same_length: bool = True,
) -> bool:
    """Validate arrays for plotting (non-empty, same length, finite values).

    Args:
        *arrays: Arrays to validate.
        check_finite: Check that all values are finite.
        check_same_length: Check that all arrays have the same length.

    Returns:
        True if validation passes, False otherwise.
    """
    if not arrays:
        return True

    if not _check_arrays_non_empty(arrays):
        return False

    if check_same_length and not _check_arrays_same_length(arrays):
        return False

    if check_finite and not _check_arrays_finite(arrays):
        return False

    return True


def clean_array(arr: np.ndarray, *, remove_nan: bool = True) -> np.ndarray:
    """Clean array by removing NaN/Inf values.

    Args:
        arr: Input array.
        remove_nan: Remove NaN and Inf values.

    Returns:
        Cleaned array.
    """
    arr_clean = np.asarray(arr, dtype=float)
    if remove_nan:
        arr_clean = arr_clean[np.isfinite(arr_clean)]
    return arr_clean


# ============================================================================
# Common Plot Elements
# ============================================================================


def add_zero_line(
    ax: Axes,
    *,
    color: str = "black",
    linewidth: float = _PLOT_LINEWIDTH_THIN,
    linestyle: str = "--",
    alpha: float = 0.5,
    **kwargs: Any,
) -> None:
    """Add horizontal line at y=0.

    Args:
        ax: Matplotlib axes.
        color: Line color.
        linewidth: Line width.
        linestyle: Line style.
        alpha: Line opacity.
        **kwargs: Additional arguments passed to axhline.
    """
    ax.axhline(0, color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha, **kwargs)


def add_confidence_bands(
    ax: Axes,
    values: np.ndarray,
    se: float,
    z_score: float = 1.96,
    *,
    color: str = "red",
    linestyle: str = "--",
    linewidth: float = 1.0,
    label: str | None = None,
) -> None:
    """Add confidence bands at ±z_score * se.

    Args:
        ax: Matplotlib axes.
        values: Central values (ignored, bands are at ±z_score*se).
        se: Standard error.
        z_score: Z-score for confidence level (1.96 for 95%).
        color: Band color.
        linestyle: Line style.
        linewidth: Line width.
        label: Label for legend (applied to upper band only).
    """
    upper = z_score * se
    lower = -z_score * se

    ax.axhline(upper, color=color, linestyle=linestyle, linewidth=linewidth, label=label)
    ax.axhline(lower, color=color, linestyle=linestyle, linewidth=linewidth)


def add_grid(ax: Axes, *, alpha: float = 0.3, linestyle: str = "--", **kwargs: Any) -> None:
    """Add grid to axes with standard settings.

    Args:
        ax: Matplotlib axes.
        alpha: Grid opacity.
        linestyle: Grid line style.
        **kwargs: Additional arguments passed to grid().
    """
    ax.grid(True, alpha=alpha, linestyle=linestyle, **kwargs)


def add_legend(
    ax: Axes,
    *,
    loc: str = "best",
    framealpha: float = 0.9,
    fontsize: int | str = 9,
    **kwargs: Any,
) -> None:
    """Add legend to axes with standard settings.

    Args:
        ax: Matplotlib axes.
        loc: Legend location.
        framealpha: Frame opacity.
        fontsize: Font size.
        **kwargs: Additional arguments passed to legend().
    """
    ax.legend(loc=loc, framealpha=framealpha, fontsize=fontsize, **kwargs)


# ============================================================================
# Styling
# ============================================================================


def setup_plot_style(style: str = "default") -> None:
    """Setup global matplotlib style settings.

    Args:
        style: Style name ('default', 'seaborn', 'ggplot', etc.).
    """
    if style != "default":
        plt.style.use(style)


def get_color_palette(n_colors: int = 10) -> list[str]:
    """Get standard color palette for consistent plotting.

    Args:
        n_colors: Number of colors to return.

    Returns:
        List of hex color codes.
    """
    # Standard matplotlib tab10 colors
    colors = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # gray
        "#bcbd22",  # olive
        "#17becf",  # cyan
    ]
    return colors[:n_colors]


# ============================================================================
# Convenience Functions
# ============================================================================


def ensure_output_directory(output_path: str | Path) -> Path:
    """Ensure output directory exists.

    Args:
        output_path: Output file path.

    Returns:
        Path object with created parent directories.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


# ============================================================================
# Statistical Plot Components
# ============================================================================


def plot_histogram_with_normal_overlay(
    ax: Axes,
    data: np.ndarray | pd.Series,
    *,
    bins: int = 50,
    show_mean_line: bool = True,
    hist_color: str = "#2E86AB",
    fit_color: str = "#A23B72",
    distribution: str = "normal",
    nu: float | None = None,
    lam: float | None = None,
    **kwargs: Any,
) -> tuple[float, float]:
    """Plot histogram with fitted theoretical distribution overlay.

    Args:
        ax: Matplotlib axes to plot on.
        data: Data to plot (array or Series).
        bins: Number of histogram bins.
        show_mean_line: Whether to add vertical line at mean.
        hist_color: Color for histogram bars.
        fit_color: Color for theoretical fit curve.
        distribution: Distribution type ('normal', 'skewt').
        nu: Degrees of freedom for Skew-t distribution.
        lam: Asymmetry parameter for Skew-t distribution.
        **kwargs: Additional arguments passed to ax.hist().

    Returns:
        Tuple of (mean, std) of the data.
    """
    from scipy import stats

    data_clean = clean_array(np.asarray(data))

    if len(data_clean) == 0:
        logger.warning("No valid data for histogram")
        return (0.0, 0.0)

    mean = float(np.mean(data_clean))
    std = float(np.std(data_clean))

    # Plot histogram
    ax.hist(
        data_clean,
        bins=bins,
        density=True,
        alpha=0.7,
        color=hist_color,
        edgecolor="black",
        linewidth=0.5,
        **kwargs,
    )

    # Overlay theoretical distribution
    x_range = np.linspace(data_clean.min(), data_clean.max(), 500)

    if distribution.lower() == "skewt" and nu is not None and lam is not None:
        # For Skew-t, use normal approximation since we don't have PDF function
        # The standardized residuals should follow Skew-t, but we approximate with normal
        # for visualization purposes
        normal_fit = stats.norm.pdf(x_range, 0, 1)  # Standardized normal
        ax.plot(
            x_range,
            normal_fit,
            color=fit_color,
            linewidth=2.5,
            label="Theoretical fit (Skew-t approx)",
        )
    else:
        # Default normal distribution
        normal_fit = stats.norm.pdf(x_range, 0, 1)  # Standardized normal
        ax.plot(x_range, normal_fit, color=fit_color, linewidth=2.5, label="Normal fit")

    # Add mean line
    if show_mean_line:
        ax.axvline(mean, color="red", linestyle="--", linewidth=1.5, alpha=0.8)

    return mean, std


def plot_qq_normal(
    ax: Axes,
    data: np.ndarray | pd.Series,
    *,
    add_reference_line: bool = True,
    show_correlation: bool = True,
    **kwargs: Any,
) -> float:
    """Plot Q-Q plot against normal distribution.

    Args:
        ax: Matplotlib axes to plot on.
        data: Data to plot (array or Series).
        add_reference_line: Whether to add 45-degree reference line.
        show_correlation: Whether to display correlation coefficient.
        **kwargs: Additional arguments passed to ax.scatter().

    Returns:
        Correlation coefficient between theoretical and sample quantiles.
    """
    from scipy import stats

    data_clean = clean_array(np.asarray(data))

    if len(data_clean) == 0:
        logger.warning("No valid data for Q-Q plot")
        return 0.0

    # Standardize data
    mean = float(np.mean(data_clean))
    std = float(np.std(data_clean))
    standardized = (data_clean - mean) / std if std > 0 else data_clean

    # Calculate quantiles
    sorted_data = cast(np.ndarray, np.sort(standardized))
    n = len(sorted_data)
    theoretical_quantiles = cast(np.ndarray, stats.norm.ppf(np.linspace(0.01, 0.99, n)))

    # Scatter plot
    ax.scatter(
        theoretical_quantiles,
        sorted_data,
        alpha=0.6,
        s=20,
        edgecolors="black",
        linewidths=0.5,
        **kwargs,
    )

    # Reference line
    if add_reference_line:
        min_val = min(theoretical_quantiles.min(), sorted_data.min())
        max_val = max(theoretical_quantiles.max(), sorted_data.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, alpha=0.8)

    # Correlation
    corr = float(np.corrcoef(theoretical_quantiles, sorted_data)[0, 1])

    if show_correlation:
        ax.text(
            0.05,
            0.95,
            f"Correlation: {corr:.4f}\n(Closer to 1 = more normal)",
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            bbox=_TEXTBOX_STYLE_DEFAULT,
        )

    return corr


def plot_series_with_train_test_split(
    ax: Axes,
    series: pd.Series,
    split_date: str | pd.Timestamp | None,
    *,
    train_color: str = "#2E86AB",
    test_color: str = "#A23B72",
    show_split_line: bool = True,
    linewidth: float = 1.0,
    alpha: float = 0.8,
    **kwargs: Any,
) -> None:
    """Plot time series with visual train/test split.

    Args:
        ax: Matplotlib axes to plot on.
        series: Time series data with datetime index.
        split_date: Date to split train/test. If None, plot entire series.
        train_color: Color for training data.
        test_color: Color for test data.
        show_split_line: Whether to add vertical line at split.
        linewidth: Line width for plot.
        alpha: Transparency for plot lines.
        **kwargs: Additional arguments passed to ax.plot().
    """
    if split_date is not None:
        split_dt = pd.to_datetime(split_date)
        train = series[series.index < split_dt]
        test = series[series.index >= split_dt]

        if not train.empty:
            ax.plot(
                train.index, train, color=train_color, linewidth=linewidth, alpha=alpha, **kwargs
            )
        if not test.empty:
            ax.plot(test.index, test, color=test_color, linewidth=linewidth, alpha=alpha, **kwargs)

        if show_split_line:
            ax.axvline(
                mdates.date2num(split_dt).item(),
                color="red",
                linestyle="--",
                linewidth=2,
                alpha=0.7,
            )
    else:
        ax.plot(series.index, series, color=train_color, linewidth=linewidth, alpha=alpha, **kwargs)


def _compute_statistics_dict(
    data_clean: np.ndarray,
    statistics: list[str],
) -> dict[str, float]:
    """Compute statistics dictionary based on requested statistics."""
    from scipy import stats as sp_stats

    stats_dict = {}

    stat_functions = {
        "mean": lambda x: float(np.mean(x)),
        "std": lambda x: float(np.std(x)),
        "skew": lambda x: float(sp_stats.skew(x)),
        "kurt": lambda x: float(sp_stats.kurtosis(x)),
        "n": lambda x: len(x),
    }

    stat_names = {
        "mean": "Mean",
        "std": "Std Dev",
        "skew": "Skewness",
        "kurt": "Kurtosis",
        "n": "N",
    }

    for stat in statistics:
        if stat in stat_functions:
            stats_dict[stat_names[stat]] = stat_functions[stat](data_clean)

    return stats_dict


def add_statistics_textbox(
    ax: Axes,
    data: np.ndarray | pd.Series,
    *,
    position: tuple[float, float] = (0.5, 1.02),
    statistics: list[str] | None = None,
    additional_stats: dict[str, float] | None = None,
    precision: int = 6,
    style: dict | None = None,
    ha: str = "center",
    va: str = "bottom",
    **text_kwargs: Any,
) -> None:
    """Add configurable statistics text box to axes.

    Args:
        ax: Matplotlib axes to add text box to.
        data: Data to compute statistics from.
        position: Text box position in axes coordinates (x, y).
        statistics: List of statistics to show. Supported: ["mean", "std", "skew", "kurt", "n"].
                   If None, shows ["mean", "std", "n"].
        additional_stats: Additional custom statistics dict.
        precision: Number of decimal places for formatting.
        style: Text box style dict. If None, uses default from constants.
        ha: Horizontal alignment.
        va: Vertical alignment.
        **text_kwargs: Additional arguments passed to ax.text().
    """
    data_clean = clean_array(np.asarray(data))

    if len(data_clean) == 0:
        return

    if statistics is None:
        statistics = ["mean", "std", "n"]

    if style is None:
        style = _TEXTBOX_STYLE_DEFAULT

    # Compute statistics
    stats_dict = _compute_statistics_dict(data_clean, statistics)

    # Add custom statistics
    if additional_stats:
        stats_dict.update(additional_stats)

    # Format text
    text_lines = []
    for key, value in stats_dict.items():
        if key == "N":
            text_lines.append(f"{key} = {value:,}")
        else:
            text_lines.append(f"{key} = {value:.{precision}f}")

    text = " | ".join(text_lines)

    ax.text(
        position[0],
        position[1],
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        bbox=style,
        **text_kwargs,
    )


def add_metrics_textbox(
    ax: Axes,
    metrics: dict[str, float],
    *,
    position: tuple[float, float] = (0.02, 0.98),
    precision: int = 8,
    style: dict | None = None,
    ha: str = "left",
    va: str = "top",
    **text_kwargs: Any,
) -> None:
    """Add metrics dictionary as formatted text box.

    Args:
        ax: Matplotlib axes to add text box to.
        metrics: Dictionary of metric names to values.
        position: Text box position in axes coordinates (x, y).
        precision: Number of decimal places for formatting.
        style: Text box style dict. If None, uses default from constants.
        ha: Horizontal alignment.
        va: Vertical alignment.
        **text_kwargs: Additional arguments passed to ax.text().
    """
    if not metrics:
        return

    if style is None:
        style = _TEXTBOX_STYLE_DEFAULT

    text_lines = [f"{key}: {value:.{precision}f}" for key, value in metrics.items()]
    text = "\n".join(text_lines)

    ax.text(
        position[0],
        position[1],
        text,
        transform=ax.transAxes,
        fontsize=10,
        ha=ha,
        va=va,
        bbox=style,
        **text_kwargs,
    )


def subsample_for_plotting(
    df: pd.DataFrame,
    max_points: int = 500,
    method: str = "uniform",
) -> pd.DataFrame:
    """Subsample dataframe for readable plots.

    Args:
        df: DataFrame to subsample.
        max_points: Maximum number of points to keep.
        method: Subsampling method ("uniform" or "random").

    Returns:
        Subsampled DataFrame.
    """
    if len(df) <= max_points:
        return df

    if method == "uniform":
        step = len(df) // max_points
        return cast(pd.DataFrame, df.iloc[::step])
    if method == "random":
        return cast(pd.DataFrame, df.sample(n=max_points, random_state=42).sort_index())

    msg = f"Unknown subsampling method: {method}"
    raise ValueError(msg)


def get_test_result_style(
    p_value: float,
    alpha: float = 0.05,
) -> Mapping[str, object]:
    """Get textbox style based on statistical test result.

    Args:
        p_value: P-value from statistical test.
        alpha: Significance level.

    Returns:
        Style dictionary for text box.
    """
    if p_value > alpha:
        return _TEXTBOX_STYLE_SUCCESS
    return _TEXTBOX_STYLE_ERROR


def load_json_if_exists(
    file_path: str | Path | None,
) -> dict | None:
    """Load JSON file if it exists, return None otherwise.

    Args:
        file_path: Path to JSON file.

    Returns:
        Loaded dictionary or None if file doesn't exist.
    """
    if file_path is None:
        return None

    json_path = Path(file_path)
    if not json_path.exists():
        return None

    try:
        import json

        with open(json_path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load JSON from {file_path}: {e}")
        return None


__all__ = [
    # Figure creation
    "create_figure_canvas",
    "create_standard_figure",
    # Saving
    "save_figure",
    "save_canvas",
    "save_plot_wrapper",
    # Date formatting
    "format_date_axis",
    "format_seasonal_axis_daily",
    # Data preparation
    "prepare_temporal_axis",
    "validate_plot_arrays",
    "clean_array",
    # Plot elements
    "add_zero_line",
    "add_confidence_bands",
    "add_grid",
    "add_legend",
    # Styling
    "setup_plot_style",
    "get_color_palette",
    # Utilities
    "ensure_output_directory",
    # Statistical plots
    "plot_histogram_with_normal_overlay",
    "plot_qq_normal",
    "plot_series_with_train_test_split",
    "add_statistics_textbox",
    "add_metrics_textbox",
    "subsample_for_plotting",
    "get_test_result_style",
    "load_json_if_exists",
]
