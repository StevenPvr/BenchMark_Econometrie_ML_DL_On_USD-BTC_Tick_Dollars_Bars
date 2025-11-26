"""Automated BTC/USD tick data fetching."""

from __future__ import annotations

from pathlib import Path
import sys

# Add project root to Python path for direct execution
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config_logging import setup_logging
from src.constants import END_DATE, EXCHANGE_ID, START_DATE, SYMBOL
from src.data_fetching.fetching import download_ticks_in_date_range
from src.path import DATASET_RAW_CSV, DATASET_RAW_PARQUET
from src.utils import get_logger

logger = get_logger(__name__)


def _validate_dependencies() -> None:
    """Validate that required dependencies are available."""
    try:
        import ccxt  # type: ignore[import-untyped]
        logger.info("✓ ccxt library available")
    except ImportError as e:
        logger.error("Missing required dependency: ccxt")
        logger.error("Install with: pip install ccxt")
        raise RuntimeError("ccxt library not available") from e

    try:
        import pandas as pd  # type: ignore[import-untyped]
        logger.info("✓ pandas library available")
    except ImportError as e:
        logger.error("Missing required dependency: pandas")
        raise RuntimeError("pandas library not available") from e


def _log_header() -> None:
    """Display the automated fetching banner with configuration."""
    logger.info("\n")
    logger.info("=" * 70)
    logger.info(" AUTOMATED BTC/USD TICK DATA FETCHING")
    logger.info("=" * 70)
    logger.info(f"Exchange:     {EXCHANGE_ID}")
    logger.info(f"Symbol:       {SYMBOL}")
    logger.info(f"Date Range:   {START_DATE} → {END_DATE}")
    logger.info(f"Output CSV:   {DATASET_RAW_CSV}")
    logger.info(f"Output PQ:    {DATASET_RAW_PARQUET}")
    logger.info("=" * 70)


def _log_footer(success: bool = True) -> None:
    """Display completion summary."""
    logger.info("\n" + "=" * 70)
    if success:
        logger.info(" FETCHING COMPLETE")
        logger.info("✓ BTC/USD tick data successfully downloaded")
    else:
        logger.info(" FETCHING FAILED")
    logger.info("=" * 70)


def main() -> None:
    """Automated main function for BTC/USD tick data fetching."""
    setup_logging()

    try:
        # Validate dependencies before proceeding
        _validate_dependencies()

        # Display header with configuration
        _log_header()

        # Execute the fetching pipeline automatically
        logger.info("Starting automated data fetching...")
        download_ticks_in_date_range()
        logger.info("✓ Data fetching completed successfully")

        # Display success footer
        _log_footer(success=True)

    except KeyboardInterrupt:
        logger.warning("Operation interrupted by user")
        _log_footer(success=False)
        sys.exit(130)
    except (FileNotFoundError, PermissionError) as e:
        logger.error("File system error: %s", e)
        logger.error("Check that output directories exist and are writable")
        _log_footer(success=False)
        sys.exit(1)
    except (ConnectionError, TimeoutError) as e:
        logger.error("Network error: %s", e)
        logger.error("Check your internet connection and exchange availability")
        _log_footer(success=False)
        sys.exit(1)
    except ValueError as e:
        logger.error("Configuration error: %s", e)
        logger.error("Check your exchange ID, symbol, and date formats")
        _log_footer(success=False)
        sys.exit(1)
    except RuntimeError as e:
        logger.error("Runtime error: %s", e)
        _log_footer(success=False)
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error during data fetching: %s", e)
        _log_footer(success=False)
        sys.exit(1)


if __name__ == "__main__":
    main()
