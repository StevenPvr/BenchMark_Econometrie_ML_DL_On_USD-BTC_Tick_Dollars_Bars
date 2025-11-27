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
import pandas as pd

logger = get_logger(__name__)


def _calculate_optimal_threshold(parquet_path: Path, target_ticks: int = 50) -> float:
    """Calculate optimal fixed threshold based on mean dollar value per tick."""
    logger.info("Loading columns 'price' and 'amount' to calculate mean dollar value...")
    # Load only necessary columns for speed
    df = pd.read_parquet(parquet_path, columns=["price", "amount"])
    
    mean_dollar_value = (df["price"] * df["amount"]).mean()
    threshold = mean_dollar_value * target_ticks
    
    logger.info("  • Mean Dollar Value/Tick: %.2f", mean_dollar_value)
    logger.info("  • Target Ticks/Bar:       %d", target_ticks)
    logger.info("  • Calculated Threshold:   %.2f", threshold)
    
    return float(threshold)


def main() -> None:
    """Main CLI function for data preparation - generates dollar bars."""
    setup_logging()

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
            
            # For partitioned data, we need an estimate. We'll load the first partition to estimate the mean.
            # This is an approximation but sufficient for setting a fixed threshold.
            first_partition = next(DATASET_CLEAN_PARQUET.glob("*.parquet"))
            logger.info("Estimating threshold from first partition: %s", first_partition.name)
            threshold = _calculate_optimal_threshold(first_partition, target_ticks=50)
            
            logger.info("Generating dollar bars from partitioned files with FIXED threshold: %.2f", threshold)
            df_bars = run_dollar_bars_pipeline_batch(
                input_dir=DATASET_CLEAN_PARQUET,
                output_parquet=DOLLAR_BARS_PARQUET,
                output_csv=DOLLAR_BARS_CSV,
                threshold=threshold,
                adaptive=False
            )
        else:
            # Single file - use traditional processing
            logger.info("📄 Detected single file - using traditional processing")
            
            logger.info("Calculating optimal fixed threshold from dataset...")
            threshold = _calculate_optimal_threshold(DATASET_CLEAN_PARQUET, target_ticks=50)
            
            logger.info("Generating dollar bars with FIXED threshold: %.2f", threshold)
            df_bars = run_dollar_bars_pipeline(
                input_parquet=DATASET_CLEAN_PARQUET,
                output_parquet=DOLLAR_BARS_PARQUET,
                output_csv=DOLLAR_BARS_CSV,
                threshold=threshold,
                adaptive=False
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

        # Add log returns (x100) into the consolidated dollar_bars dataset
        logger.info("=" * 60)
        logger.info("Adding log returns (x100) into dollar_bars dataset...")
        add_log_returns_to_bars_file(DOLLAR_BARS_PARQUET, DOLLAR_BARS_CSV)

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
