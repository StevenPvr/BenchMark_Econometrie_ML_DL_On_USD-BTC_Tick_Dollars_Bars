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
from src.data_preparation.preparation import (
    add_log_returns_to_bars_file,
    run_dollar_bars_pipeline,
    run_dollar_bars_pipeline_batch,
)
from src.path import (
    DATASET_CLEAN_PARQUET,
    DOLLAR_BARS_PARQUET,
    DOLLAR_BARS_CSV,
)
from src.utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Main CLI function for data preparation - generates dollar bars."""
    setup_logging()

    calibration_fraction = 0.2  # use first 20% of ticks to calibrate T_0 (no lookahead)

    logger.info("Starting data preparation pipeline")
    logger.info("=" * 60)
    logger.info("DATA PREPARATION: Dollar Bars Generation")
    logger.info("=" * 60)

    try:
        # Check if input data exists
        if not DATASET_CLEAN_PARQUET.exists():
            logger.error("Input data not found: %s", DATASET_CLEAN_PARQUET)
            logger.error("Please run data_fetching and data_cleaning first")
            sys.exit(1)

        # Determine processing approach based on input type
        if DATASET_CLEAN_PARQUET.is_dir():
            # Directory with partitioned files - use batch processing
            logger.info("📁 Detected partitioned dataset - using memory-efficient batch processing")
            logger.info("Generating dollar bars (De Prado fixed threshold, prefix calibration)...")
            df_bars = run_dollar_bars_pipeline_batch(
                input_dir=DATASET_CLEAN_PARQUET,
                target_num_bars=500_000,
                output_parquet=DOLLAR_BARS_PARQUET,
                adaptive=False,
                calibration_fraction=calibration_fraction,
                include_incomplete_final=True,
            )
        else:
            # Single file - use traditional processing
            logger.info("📄 Detected single file - using traditional processing")
            logger.info("Generating dollar bars (De Prado fixed threshold, prefix calibration)...")
            df_bars = run_dollar_bars_pipeline(
                target_num_bars=500_000,
                input_parquet=DATASET_CLEAN_PARQUET,
                output_parquet=DOLLAR_BARS_PARQUET,
                adaptive=False,
                calibration_fraction=calibration_fraction,
                include_incomplete_final=True,
            )

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

        # Add log returns (natural) into the consolidated dollar_bars dataset
        logger.info("=" * 60)
        logger.info("Adding log returns (natural) into dollar_bars dataset...")
        add_log_returns_to_bars_file(DOLLAR_BARS_PARQUET)

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
