"""Data preparation functions for ARIMA models.

Provides thin wrappers that honor module-level default paths so tests can
monkeypatch `src.data_preparation.data_preparation.*` constants to tmp files.
"""

from __future__ import annotations

from src.constants import TRAIN_RATIO_DEFAULT
from src.data_preparation.ticker_preparation import (
    add_log_returns_to_ticker_parquet,
    split_tickers_train_test,
)
from src.data_preparation.timeseriessplit import (
    load_train_test_data as _ts_load_train_test_data,
    split_train_test as _ts_split_train_test,
)
from src.path import (
    WEIGHTED_LOG_RETURNS_FILE as DEFAULT_RETURNS_FILE,
    WEIGHTED_LOG_RETURNS_SPLIT_FILE as DEFAULT_SPLIT_FILE,
)

# Module-level defaults (overridable in tests via monkeypatch)
WEIGHTED_LOG_RETURNS_FILE = DEFAULT_RETURNS_FILE
WEIGHTED_LOG_RETURNS_SPLIT_FILE = DEFAULT_SPLIT_FILE


def split_train_test(
    train_ratio: float = TRAIN_RATIO_DEFAULT,
    input_file: str | None = None,
    output_file: str | None = None,
) -> None:
    """Wrapper that forwards patched default paths to the timeseriessplit module."""
    if input_file is None:
        input_file = str(WEIGHTED_LOG_RETURNS_FILE)
    if output_file is None:
        output_file = str(WEIGHTED_LOG_RETURNS_SPLIT_FILE)
    _ts_split_train_test(train_ratio=train_ratio, input_file=input_file, output_file=output_file)


def load_train_test_data(input_file: str | None = None):  # type: ignore[override]
    """Wrapper honoring patched default split path for loading train/test series."""
    if input_file is None:
        input_file = str(WEIGHTED_LOG_RETURNS_SPLIT_FILE)
    return _ts_load_train_test_data(input_file=input_file)


__all__ = [
    "load_train_test_data",
    "split_train_test",
    "split_tickers_train_test",
    "add_log_returns_to_ticker_parquet",
]
