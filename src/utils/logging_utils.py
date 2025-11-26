"""Logging utilities for data processing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import pandas as pd  # type: ignore[import-untyped]

from src.utils.io import ensure_output_dir


def log_series_summary(series: pd.Series, name: str = "Series") -> None:
    """Log summary statistics for a pandas Series."""
    from src.config_logging import get_logger
    logger = get_logger(__name__)

    logger.info(f"{name} summary:")
    logger.info(f"  Length: {len(series)}")
    logger.info(f"  Null values: {series.isnull().sum()}")
    logger.info(f"  Mean: {series.mean():.6f}")
    logger.info(f"  Std: {series.std():.6f}")
    logger.info(f"  Min: {series.min()}")
    logger.info(f"  Max: {series.max()}")


def log_split_summary(
    train_data: pd.DataFrame | pd.Series,
    test_data: pd.DataFrame | pd.Series,
    split_date: Any = None,
) -> None:
    """Log summary of train/test split."""
    from src.config_logging import get_logger
    logger = get_logger(__name__)

    logger.info("Train/Test split summary:")
    logger.info(f"  Train size: {len(train_data)}")
    logger.info(f"  Test size: {len(test_data)}")
    logger.info(".1%")
    if split_date is not None:
        logger.info(f"  Split date: {split_date}")


def save_plot(figure: Any, path: Path | str, dpi: int = 300) -> None:
    """Save matplotlib figure to file."""
    ensure_output_dir(path)
    figure.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(figure)
