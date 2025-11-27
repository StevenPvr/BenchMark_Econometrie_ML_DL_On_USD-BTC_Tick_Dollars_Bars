"""Hyperparameter optimization for Ridge Classifier on the train split only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).parent.parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config_logging import get_logger, setup_logging
from src.model.econometrie.ridge.main import load_data
from src.model.econometrie.ridge.pipeline import (
    RidgePipeline,
    RidgePipelineConfig,
)
from src.optimisation import save_optimization_results
from src.path import (
    RIDGE_CLASSIFIER_ARTIFACTS_DIR,
    RIDGE_CLASSIFIER_BEST_PARAMS_FILE,
)
from src.utils import save_json_pretty

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Ridge hyperparameter search."""
    parser = argparse.ArgumentParser(description="Optimize Ridge hyperparameters on train split")
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of Optuna trials (default: 100)",
    )
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
    return parser.parse_args()


def main() -> None:
    """Run hyperparameter optimization on the train split."""
    args = parse_args()
    setup_logging()

    logger.info("=" * 60)
    logger.info("RIDGE CLASSIFIER HYPERPARAMETER OPTIMIZATION (train split)")
    logger.info("=" * 60)

    # Load data (train only for search)
    X_train, _, y_train, _ = load_data()

    RIDGE_CLASSIFIER_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    config = RidgePipelineConfig(
        n_trials=args.n_trials,
        n_splits=args.n_splits,
        purge_gap=args.purge_gap,
        min_train_size=100,
        output_dir=RIDGE_CLASSIFIER_ARTIFACTS_DIR,
        verbose=True,
    )

    pipeline = RidgePipeline(config)
    opt_result = pipeline.optimize(X_train, y_train)

    # Persist best params and brief summary
    save_json_pretty(opt_result.best_params, RIDGE_CLASSIFIER_BEST_PARAMS_FILE)
    summary_path = RIDGE_CLASSIFIER_ARTIFACTS_DIR / "optimization_results.json"
    save_optimization_results(opt_result, summary_path)

    logger.info("Optimization complete. Best params saved to %s", RIDGE_CLASSIFIER_BEST_PARAMS_FILE)
    logger.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
