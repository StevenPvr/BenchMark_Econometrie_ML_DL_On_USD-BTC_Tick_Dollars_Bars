"""Train Lasso Classifier on the train split only and save artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).parent.parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config_logging import get_logger, setup_logging
from src.model.econometrie.lasso.main import load_data
from src.model.econometrie.lasso.pipeline import (
    LassoPipeline,
    LassoPipelineConfig,
)
from src.path import (
    LASSO_CLASSIFIER_ARTIFACTS_DIR,
    LASSO_CLASSIFIER_BEST_PARAMS_FILE,
    LASSO_CLASSIFIER_MODEL_FILE,
    LASSO_CLASSIFIER_TRAINING_RESULTS_FILE,
)
from src.training.trainer import save_training_results
from src.utils import save_json_pretty

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Lasso Classifier training."""
    parser = argparse.ArgumentParser(description="Train Lasso Classifier on train split only")
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
        default=LASSO_CLASSIFIER_BEST_PARAMS_FILE,
        help=f"Path to best params JSON (default: {LASSO_CLASSIFIER_BEST_PARAMS_FILE})",
    )
    return parser.parse_args()


def _load_params(params_path: Path) -> dict:
    """Load hyperparameters from JSON file."""
    import json

    with params_path.open("r") as f:
        return json.load(f)


def main() -> None:
    """Run Lasso Classifier training on the train split."""
    args = parse_args()
    setup_logging()

    logger.info("=" * 60)
    logger.info("LASSO CLASSIFIER TRAIN (train split only)")
    logger.info("=" * 60)

    # Load data and keep test split untouched
    X_train, _, y_train, _ = load_data()

    LASSO_CLASSIFIER_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load best params
    params_path = args.params
    if not params_path.exists():
        raise FileNotFoundError(
            f"Best params file not found at {params_path}. "
            "Run the optimization script first."
        )
    best_params = _load_params(params_path)

    config = LassoPipelineConfig(
        n_splits=args.n_splits,
        purge_gap=args.purge_gap,
        min_train_size=100,
        output_dir=LASSO_CLASSIFIER_ARTIFACTS_DIR,
        verbose=True,
        validation_split=0.0,  # No validation split
    )

    pipeline = LassoPipeline(config)

    # Train on full train split only (no CV, no validation)
    model, train_result = pipeline.train(X_train, y_train, best_params, use_cv=False)

    # Save artifacts
    model.save(LASSO_CLASSIFIER_MODEL_FILE)
    save_json_pretty(best_params, LASSO_CLASSIFIER_BEST_PARAMS_FILE)
    save_training_results(train_result, LASSO_CLASSIFIER_TRAINING_RESULTS_FILE)

    # Log feature selection info (multi-class: coefficients are 2D)
    coefs = model.coef_
    if coefs.ndim == 1:
        n_selected = sum(1 for c in coefs if c != 0)
        n_total = len(coefs)
    else:
        n_selected = sum(1 for i in range(coefs.shape[1]) if any(coefs[:, i] != 0))
        n_total = coefs.shape[1]
    logger.info("Training complete. Model saved to %s", LASSO_CLASSIFIER_MODEL_FILE)
    logger.info("Training metrics saved to %s", LASSO_CLASSIFIER_TRAINING_RESULTS_FILE)
    logger.info("Selected features: %d/%d", n_selected, n_total)


if __name__ == "__main__":
    main()
