"""Main pipeline for feature generation.

This module orchestrates the complete feature engineering pipeline:

1. Load input: log_returns_split.parquet
2. Compute all features (microstructure, volatility, momentum, etc.)
3. Apply intelligent lag structure
4. Split into train/test
5. Save raw (unscaled) outputs:
   - dataset_features.parquet: Raw features for tree-based ML (XGBoost, etc.)
   - dataset_features_linear.parquet: Copy for linear models (scaled in clear_features)
   - dataset_features_lstm.parquet: Copy for LSTM (scaled in clear_features)

NOTE: Scaler fitting moved to clear_features module (after PCA transformation).

Usage:
    python -m src.features.main
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add project root to Python path for direct execution.
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

NUMBA_CACHE_DIR = os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
Path(NUMBA_CACHE_DIR).mkdir(parents=True, exist_ok=True)

import pandas as pd  # type: ignore[import-untyped]

from src.config_logging import setup_logging
from src.utils import get_logger
from src.features.batch import (
    BATCH_OVERLAP,
    DEFAULT_BATCH_SIZE,
    run_batch_pipeline,
)
from src.features.compute import apply_lags, compute_all_features
from src.features.io import load_input_data, save_outputs
from src.features.pipeline import (
    TRAIN_RATIO,
    compute_timestamp_features,
    drop_initial_nan_rows,
    interpolate_sporadic_nan,
    shift_target_to_future_return,
    split_train_test,
)
from src.path import (
    DATASET_FEATURES_LINEAR_PARQUET,
    DATASET_FEATURES_LSTM_PARQUET,
    DATASET_FEATURES_PARQUET,
)

logger = get_logger(__name__)


def main() -> None:
    """Run the complete feature engineering pipeline.

    Pipeline flow:
    1. Load dollar bars
    2. Compute all features + lags
    3. Clean NaN rows
    4. Split train/test
    5. Save raw datasets (unscaled)

    Uses batch mode by default to handle large datasets efficiently.

    Next step: Run clear_features to apply PCA, fit scalers, and normalize.
    """
    parser = argparse.ArgumentParser(
        description="Feature Engineering Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for processing (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=BATCH_OVERLAP,
        help=f"Overlap between batches for rolling windows (default: {BATCH_OVERLAP})",
    )
    parser.add_argument(
        "--no-batch",
        action="store_true",
        help="Disable batch mode (load all in memory - may crash on large datasets)",
    )
    parser.add_argument(
        "--sample",
        type=float,
        default=1.0,
        help="Sample fraction of input data (0.0-1.0). Use 0.05 for 5%% sample for testing.",
    )
    args = parser.parse_args()

    setup_logging()

    try:
        if not args.no_batch:
            run_batch_pipeline(
                batch_size=args.batch_size,
                overlap=args.overlap,
                sample_fraction=args.sample,
            )
        else:
            _run_standard_pipeline()

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        sys.exit(1)


def _run_standard_pipeline() -> None:
    """Run standard (non-batch) pipeline for small datasets."""
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING PIPELINE (NON-BATCH MODE)")
    logger.info("=" * 60)
    logger.warning("Non-batch mode may crash on large datasets!")

    # 1. Load input data
    df_bars = load_input_data()

    # 2. Compute timestamp-derived features
    df_bars = compute_timestamp_features(df_bars)

    # 3. Compute all features
    df_features = compute_all_features(df_bars)

    # 4. Apply intelligent lag structure
    df_features_lagged = apply_lags(df_features)

    # 5. Drop initial NaN rows from lags/rolling windows
    df_features_clean = drop_initial_nan_rows(df_features_lagged)

    # 6. Interpolate sporadic NaN
    df_features_clean = interpolate_sporadic_nan(df_features_clean)

    # 7. Shift target to next-bar return
    df_features_clean = shift_target_to_future_return(
        df_features_clean, target_col="log_return"
    )

    # 8. Split into train/test
    df_train, df_test, _ = split_train_test(df_features_clean)
    df_train["split"] = "train"
    df_test["split"] = "test"
    df_features_with_split = pd.concat([df_train, df_test], axis=0)

    # 9. Save raw datasets (all 3 copies unscaled)
    save_outputs(df_features_with_split)

    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING COMPLETE")
    logger.info("=" * 60)
    logger.info("Output files (all unscaled):")
    logger.info("  - %s (Raw features for tree-based ML)", DATASET_FEATURES_PARQUET)
    logger.info("  - %s (Copy for linear models)", DATASET_FEATURES_LINEAR_PARQUET)
    logger.info("  - %s (Copy for LSTM)", DATASET_FEATURES_LSTM_PARQUET)
    logger.info("")
    logger.info("NEXT STEP: Run clear_features to apply:")
    logger.info("  1. PCA reduction on correlated features")
    logger.info("  2. Scaler fitting on PCA-transformed features (train only)")
    logger.info("  3. Normalization (z-score for linear, minmax for LSTM)")


if __name__ == "__main__":
    main()
