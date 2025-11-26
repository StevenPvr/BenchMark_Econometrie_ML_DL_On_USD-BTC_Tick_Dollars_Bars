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
from src.data_preparation.preparation import run_dollar_bars_pipeline, save_log_returns_split
from src.path import DATASET_RAW_PARQUET, DOLLAR_BARS_PARQUET, DOLLAR_BARS_CSV, WEIGHTED_LOG_RETURNS_SPLIT_FILE
from src.utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Main CLI function for data preparation - generates dollar bars."""
    setup_logging()

    logger.info("Starting data preparation pipeline")
    logger.info("=" * 60)
    logger.info("DATA PREPARATION: Dollar Bars Generation")
    logger.info("=" * 60)

    try:
        # Check if input data exists
        if not DATASET_RAW_PARQUET.exists():
            logger.error("Input data not found: %s", DATASET_RAW_PARQUET)
            logger.error("Please run data_fetching and data_cleaning first")
            sys.exit(1)

        # Generate dollar bars using adaptive threshold (De Prado methodology)
        logger.info("Generating dollar bars with adaptive threshold...")
        df_bars = run_dollar_bars_pipeline()

        logger.info("✓ Dollar bars generation completed")
        logger.info("  • Bars generated: %d", len(df_bars))
        logger.info("  • Output saved to:")
        logger.info("    - %s", DOLLAR_BARS_PARQUET)
        logger.info("    - %s", DOLLAR_BARS_CSV)

        # Log basic statistics
        if not df_bars.empty and 'close' in df_bars.columns:
            logger.info("  • Price range: %.2f - %.2f",
                       df_bars['close'].min(), df_bars['close'].max())
            logger.info("  • Date range: %s - %s",
                       df_bars.index.min(), df_bars.index.max())

        # Generate log returns split (x100) for GARCH
        logger.info("=" * 60)
        logger.info("Generating log returns (x100) for GARCH...")
        log_returns_df = save_log_returns_split(df_bars)
        logger.info("  • Log returns generated: %d", len(log_returns_df))
        logger.info("  • Output saved to: %s", WEIGHTED_LOG_RETURNS_SPLIT_FILE)

        logger.info("=" * 60)
        logger.info("Data preparation completed successfully")
        logger.info("Next step: Run ARIMA training (python -m src.arima.training_arima.main)")

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        logger.error("Ensure data_fetching and data_cleaning have been run first")
        sys.exit(1)
    except ValueError as e:
        logger.error("Data validation error: %s", e)
        logger.error("Check input data quality")
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error during data preparation: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
