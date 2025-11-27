"""Train Logistic Regression on the train split only and save artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).parent.parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config_logging import get_logger, setup_logging
from src.model.econometrie.ols.main import load_data
from src.model.econometrie.ols.pipeline import (
    LogisticPipeline,
    LogisticPipelineConfig,
)
from src.path import (
    LOGISTIC_ARTIFACTS_DIR,
    LOGISTIC_BEST_PARAMS_FILE,
    LOGISTIC_MODEL_FILE,
    LOGISTIC_TRAINING_RESULTS_FILE,
)
from src.training.trainer import save_training_results
from src.utils import save_json_pretty

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Logistic Regression training."""
    parser = argparse.ArgumentParser(description="Train Logistic Regression on train split only")
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of CV splits for optimization (default: 5)",
    )
    parser.add_argument(
        "--purge-gap",
        type=int,
        default=5,
        help="Purge gap between train/test in CV (default: 5)",
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=LOGISTIC_BEST_PARAMS_FILE,
        help=f"Path to best params JSON (default: {LOGISTIC_BEST_PARAMS_FILE})",
    )
    return parser.parse_args()


def _load_params(params_path: Path) -> dict:
    """Load hyperparameters from JSON file."""
    import json

    with params_path.open("r") as f:
        return json.load(f)


def main() -> None:
    """Run Logistic Regression training on the train split."""
    args = parse_args()
    setup_logging()

    logger.info("=" * 60)
    logger.info("LOGISTIC REGRESSION TRAIN (train split only)")
    logger.info("=" * 60)

    # Load data and keep test split untouched
    X_train, _, y_train, _ = load_data()

    LOGISTIC_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load best params (or use defaults if file doesn't exist)
    params_path = args.params
    if params_path.exists():
        best_params = _load_params(params_path)
    else:
        logger.warning("Best params file not found at %s. Using defaults.", params_path)
        best_params = {"fit_intercept": True, "normalize": True}

    config = LogisticPipelineConfig(
        n_splits=args.n_splits,
        purge_gap=args.purge_gap,
        min_train_size=100,
        output_dir=LOGISTIC_ARTIFACTS_DIR,
        verbose=True,
        validation_split=0.0,  # No validation split
    )

    pipeline = LogisticPipeline(config)

    # Train on full train split only (no CV, no validation)
    model, train_result = pipeline.train(X_train, y_train, best_params, use_cv=False)

    # Save artifacts
    model.save(LOGISTIC_MODEL_FILE)
    save_json_pretty(best_params, LOGISTIC_BEST_PARAMS_FILE)
    save_training_results(train_result, LOGISTIC_TRAINING_RESULTS_FILE)

    logger.info("Training complete. Model saved to %s", LOGISTIC_MODEL_FILE)
    logger.info("Training metrics saved to %s", LOGISTIC_TRAINING_RESULTS_FILE)


if __name__ == "__main__":
    main()
