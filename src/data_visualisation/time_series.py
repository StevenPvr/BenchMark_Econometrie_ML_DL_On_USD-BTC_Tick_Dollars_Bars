"""Time series plotting utilities for log-returns analysis."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from src.constants import FONTSIZE_TITLE, PLOT_ALPHA_DEFAULT, PLOT_ALPHA_LIGHT
from src.utils import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

__all__ = [
    "plot_log_returns_time_series",
]

# Time series plot constants
_FIGSIZE_TIME_SERIES: tuple[int, int] = (14, 8)
_ROLLING_VOL_WINDOW: int = 20


def plot_log_returns_time_series(
    df: pd.DataFrame,
    log_returns: pd.Series,
    datetime_col: str = "datetime_close",
    output_dir: Path | None = None,
    figsize: tuple[int, int] = _FIGSIZE_TIME_SERIES,
) -> Figure:
    """
    Plot the time series of log-returns with volatility.

    Creates a two-panel figure showing:
    - Top: Log-returns time series
    - Bottom: Rolling realized volatility (standard deviation)

    Args:
        df: DataFrame with dollar bars data (for datetime column).
        log_returns: Series of log-returns.
        datetime_col: Name of the datetime column.
        output_dir: Directory to save the plot.
        figsize: Figure size as (width, height).

    Returns:
        Matplotlib Figure object.
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Get datetime index
    if datetime_col in df.columns:
        x = pd.to_datetime(df[datetime_col])
    else:
        x = df.index

    # Log-returns plot
    ax1 = axes[0]
    ax1.plot(x, log_returns, linewidth=0.5, color="steelblue", alpha=PLOT_ALPHA_DEFAULT)
    ax1.axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax1.set_ylabel("Log-Return", fontsize=10)
    ax1.set_title("Serie Temporelle des Log-Returns", fontsize=FONTSIZE_TITLE, fontweight="bold")
    ax1.grid(True, alpha=PLOT_ALPHA_LIGHT)

    # Rolling volatility plot
    ax2 = axes[1]
    rolling_vol = log_returns.rolling(window=_ROLLING_VOL_WINDOW).std()
    ax2.plot(x, rolling_vol, linewidth=1, color="orange", alpha=PLOT_ALPHA_DEFAULT)
    ax2.fill_between(x, 0, rolling_vol, alpha=PLOT_ALPHA_LIGHT, color="orange")
    ax2.set_ylabel(f"Volatilite (Std {_ROLLING_VOL_WINDOW})", fontsize=10)
    ax2.set_xlabel("Date", fontsize=10)
    ax2.set_title("Volatilite Realisee (Rolling Std)", fontsize=FONTSIZE_TITLE, fontweight="bold")
    ax2.grid(True, alpha=PLOT_ALPHA_LIGHT)

    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / "log_returns_time_series.png"
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        logger.info("Time series plot saved: %s", filepath)

    return fig
