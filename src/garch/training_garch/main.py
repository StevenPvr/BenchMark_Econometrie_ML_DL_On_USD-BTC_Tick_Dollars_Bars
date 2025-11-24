from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure project root before any src imports
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.garch.training_garch.orchestration import load_garch_dataset
from src.garch.training_garch.training import train_egarch_from_dataset
from src.utils import get_logger

logger = get_logger(__name__)


def run_training() -> dict[str, object]:
    """Stage 2: Run final training on TRAIN data with periodic refit.

    Loads optimized hyperparameters and performs final training with periodic
    refit on TRAIN data. Saves trained model and outputs.

    Returns:
        Dictionary with training info and output file paths.
    """
    logger.info("=" * 60)
    logger.info("STAGE 2: GARCH TRAINING - Final Training on TRAIN")
    logger.info("=" * 60)

    # Load dataset
    try:
        df = load_garch_dataset()
    except (FileNotFoundError, ValueError) as ex:
        logger.error("Failed to load GARCH dataset: %s", ex)
        raise

    # Train model
    try:
        info = train_egarch_from_dataset(df)
    except (FileNotFoundError, ValueError) as ex:
        logger.error("Failed to train GARCH model: %s", ex)
        raise

    # Print summary
    logger.info("=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info("Distribution: %s", info["dist"])
    logger.info("Optimized hyperparameters: %s", info["hyperparams"])
    logger.info("Training samples: %d", info["n_train"])
    logger.info("Window type: %s", info["window_type"])
    if info["window_size"] is not None:
        logger.info("Window size: %d", info["window_size"])
    logger.info("Refit frequency: %d", info["refit_freq"])
    logger.info("Number of refits: %d", info["n_refits"])
    logger.info("Standardized residuals diagnostics: %s", info["std_resid_diagnostics"])
    logger.info("")
    logger.info("Output files:")
    logger.info("  Model file: %s", info["model_file"])
    logger.info("  Metadata file: %s", info["metadata_file"])
    logger.info("  Outputs file: %s", info["outputs_file"])
    logger.info("=" * 60)

    return info


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description="GARCH training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "stage",
        nargs="?",
        default="training",
        choices=["training"],
        help="Run training (default: training)",
    )
    return parser


def main() -> None:
    """Main entry point for GARCH training pipeline."""
    create_parser().parse_args()

    try:
        run_training()
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
