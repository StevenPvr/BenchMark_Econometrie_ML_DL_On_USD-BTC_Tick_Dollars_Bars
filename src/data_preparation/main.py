"""CLI for data preparation module."""

from __future__ import annotations

from pathlib import Path
import sys

# Add project root to Python path for direct execution.
# This must be done before importing src modules.
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config_logging import setup_logging
from src.data_preparation.data_preparation import (
    load_train_test_data,
    split_tickers_train_test,
    split_train_test,
)
from src.utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Main CLI function for data preparation."""
    setup_logging()

    # Split weighted log returns (for ARIMA/GARCH)
    logger.info("Starting split of weighted log returns (train/test)")
    split_train_test()
    train_series, test_series = load_train_test_data()
    logger.info(
        "Data preparation complete: train=%d, test=%d",
        len(train_series),
        len(test_series),
    )

    # Split ticker data (for LightGBM / RF)
    logger.info("Starting split of ticker data (train/test) for RF/LightGBM")
    split_tickers_train_test()
    logger.info("Ticker data split complete")
    logger.info(
        "Note: data_tickers_full_insights.parquet will be created by rolling_garch pipeline"
    )


if __name__ == "__main__":
    main()
