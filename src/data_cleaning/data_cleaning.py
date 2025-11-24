"""High-level data cleaning orchestration for the S&P 500 dataset.

This module provides one main entry point:

- ``filter_by_membership``: apply integrity fixes and save a cleaned dataset.

The implementation fills missing data with zeros and ensures complete time series
for all tickers, rather than removing incomplete tickers.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

from src.data_cleaning import validation
from src.data_cleaning.filtering import write_filtered_dataset
from src.data_cleaning.integrity import apply_basic_integrity_fixes
from src.data_cleaning.validation import load_dataset
from src.path import DATASET_FILTERED_FILE
from src.utils import get_logger

logger = get_logger(__name__)

# Module reference for monkeypatching support in tests
_current_module = sys.modules[__name__]


def _load_dataset_with_monkeypatch() -> pd.DataFrame:
    """Load the dataset, honoring a monkeypatched DATASET_FILE if present.

    Tests can set ``src.data_cleaning.data_cleaning.DATASET_FILE`` to point to
    a temporary file. This helper mirrors that override into the validation
    module before calling :func:`load_dataset`.

    Returns:
        Loaded DataFrame with monkeypatch support.
    """
    dataset_file = getattr(_current_module, "DATASET_FILE", None)
    if dataset_file is not None:
        # Propagate override so that validation.load_dataset uses the same path
        validation.DATASET_FILE = dataset_file  # type: ignore[attr-defined]

    return load_dataset()


def filter_by_membership() -> None:
    """Apply basic integrity fixes to dataset and save output.

    **Important**: This function fills missing data with zeros and ensures
    complete time series for all tickers. It is safe to use before train/test split.

    The function:

    - loads and validates the raw dataset,
    - applies basic integrity fixes (duplicate removal, filling missing data with zeros,
      filling missing dates for incomplete tickers),
    - saves the cleaned dataset to ``DATASET_FILTERED_FILE``.

    Raises:
        FileNotFoundError: If dataset file does not exist.
        KeyError: If required columns are missing in dataset.
        OSError: If filtered dataset cannot be saved.
    """
    logger.info("Starting dataset integrity fixes and completion")

    raw_df = _load_dataset_with_monkeypatch()

    # Apply basic integrity fixes and persist
    fixed_df, counters = apply_basic_integrity_fixes(raw_df)
    dup_removed = counters.get("duplicates_removed", 0)
    missing_filled = counters.get("missing_values_filled", 0)
    dates_filled = counters.get("missing_dates_filled", 0)
    if dup_removed > 0 or missing_filled > 0 or dates_filled > 0:
        logger.info(
            "Integrity fixes: duplicates_removed=%d, missing_values_filled=%d, "
            "missing_dates_filled=%d",
            dup_removed,
            missing_filled,
            dates_filled,
        )

    # Honour potential monkeypatched output path
    output_file: Path = getattr(_current_module, "DATASET_FILTERED_FILE", DATASET_FILTERED_FILE)
    write_filtered_dataset(fixed_df, output_file=output_file)
