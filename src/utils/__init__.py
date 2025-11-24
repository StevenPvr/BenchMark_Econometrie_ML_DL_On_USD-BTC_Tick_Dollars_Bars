"""Utility functions for data processing, validation, I/O, and financial computations.

This package provides modular utilities organized by functionality:
- validation: DataFrame, file, and parameter validation
- temporal: Temporal validation and train/test splitting
- io: File I/O operations (CSV, Parquet, JSON)
- transforms: Data transformations
- datetime_utils: DateTime parsing and manipulation
- metrics: Statistical and financial metrics
- logging_utils: Logging and plotting utilities
- financial: Financial computations (liquidity weights)
"""

from __future__ import annotations

from pathlib import Path
import sys

# Import get_logger from config_logging for backward compatibility
from src.config_logging import get_logger


def setup_project_path() -> None:
    """Set up project root path for direct execution of modules.

    This function adds the project root to sys.path when modules are run directly.
    Should be called at the beginning of main functions in executable modules.
    """
    script_dir = Path(__file__).parent.parent
    project_root = script_dir.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


# DateTime utilities
from src.utils.datetime_utils import (
    extract_date_range,
    filter_by_date_range,
    format_dates_to_string,
    normalize_timestamp_to_datetime,
    parse_date_value,
)

# Financial utilities
from src.utils.financial import compute_rolling_liquidity_weights

# I/O utilities
from src.utils.io import (
    ensure_output_dir,
    get_parquet_path,
    load_and_validate_dataframe,
    load_csv_file,
    load_dataframe,
    load_json_data,
    load_parquet_file,
    read_dataset_file,
    save_json_pretty,
    save_parquet_and_csv,
    write_placeholder_file,
)

# Logging utilities
from src.utils.logging_utils import log_series_summary, log_split_summary, save_plot

# Metrics utilities
from src.utils.metrics import chi2_sf, compute_log_returns, compute_residuals

# Statsmodels utilities
from src.utils.statsmodels_utils import suppress_statsmodels_warnings

# Temporal utilities
from src.utils.temporal import (
    compute_timeseries_split_indices,
    log_split_dates,
    validate_temporal_order_series,
    validate_temporal_split,
)

# Transform utilities
from src.utils.transforms import (
    extract_features_and_target,
    filter_by_split,
    remove_metadata_columns,
    stable_ticker_id,
)

# Validation utilities
from src.utils.validation import (
    has_both_splits,
    validate_dataframe_not_empty,
    validate_file_exists,
    validate_required_columns,
    validate_series,
    validate_ticker_id,
    validate_train_ratio,
)

__all__ = [
    # Project setup
    "setup_project_path",
    # get_logger from config_logging
    "get_logger",
    # Validation
    "has_both_splits",
    "validate_dataframe_not_empty",
    "validate_file_exists",
    "validate_required_columns",
    "validate_series",
    "validate_ticker_id",
    "validate_train_ratio",
    # Temporal
    "compute_timeseries_split_indices",
    "log_split_dates",
    "validate_temporal_order_series",
    "validate_temporal_split",
    # I/O
    "ensure_output_dir",
    "get_parquet_path",
    "load_and_validate_dataframe",
    "load_csv_file",
    "load_dataframe",
    "load_json_data",
    "load_parquet_file",
    "read_dataset_file",
    "save_json_pretty",
    "save_parquet_and_csv",
    "write_placeholder_file",
    # Transforms
    "extract_features_and_target",
    "filter_by_split",
    "remove_metadata_columns",
    "stable_ticker_id",
    # DateTime
    "extract_date_range",
    "filter_by_date_range",
    "format_dates_to_string",
    "normalize_timestamp_to_datetime",
    "parse_date_value",
    # Metrics
    "chi2_sf",
    "compute_log_returns",
    "compute_residuals",
    # Logging
    "log_series_summary",
    "log_split_summary",
    "save_plot",
    # Statsmodels
    "suppress_statsmodels_warnings",
    # Financial
    "compute_rolling_liquidity_weights",
]
