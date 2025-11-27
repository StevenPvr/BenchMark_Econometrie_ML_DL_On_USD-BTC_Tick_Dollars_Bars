"""Train LightGBM on the train split only and save artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).parent.parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config_logging import get_logger, setup_logging
from src.model.machine_learning.lightgbm.main import load_data
from src.model.machine_learning.lightgbm.pipeline import (
    LightGBMPipeline,
    LightGBMPipelineConfig,
)
from src.path import (
    LIGHTGBM_ARTIFACTS_DIR,
    LIGHTGBM_BEST_PARAMS_FILE,
    LIGHTGBM_MODEL_FILE,
    LIGHTGBM_TRAINING_RESULTS_FILE,
)
from src.training.trainer import save_training_results
from src.utils import save_json_pretty

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for LightGBM training."""
    parser = argparse.ArgumentParser(description="Train LightGBM on train split only")
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
        default=LIGHTGBM_BEST_PARAMS_FILE,
        help=f"Path to best params JSON (default: {LIGHTGBM_BEST_PARAMS_FILE})",
    )
    return parser.parse_args()


def _load_params(params_path: Path) -> dict:
    """Load hyperparameters from JSON file."""
    import json

    with params_path.open("r") as f:
        return json.load(f)


def main() -> None:
    """Run LightGBM training on the train split."""
    args = parse_args()
    setup_logging()

    logger.info("=" * 60)
    logger.info("LIGHTGBM TRAIN (train split only)")
    logger.info("=" * 60)

    # Load data and keep test split untouched
    X_train, _, y_train, _ = load_data()

    LIGHTGBM_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load best params
    params_path = args.params
    if not params_path.exists():
        raise FileNotFoundError(
            f"Best params file not found at {params_path}. "
            "Run the optimization script first."
        )
    best_params = _load_params(params_path)

    config = LightGBMPipelineConfig(
        n_splits=args.n_splits,
        purge_gap=args.purge_gap,
        min_train_size=200,
        output_dir=LIGHTGBM_ARTIFACTS_DIR,
        verbose=True,
        validation_split=0.0,  # No validation split
    )

    pipeline = LightGBMPipeline(config)

    # Train on full train split only (no CV, no validation)
    model, train_result = pipeline.train(X_train, y_train, best_params, use_cv=False)

    # Save artifacts
    model.save(LIGHTGBM_MODEL_FILE)
    save_json_pretty(best_params, LIGHTGBM_BEST_PARAMS_FILE)
    save_training_results(train_result, LIGHTGBM_TRAINING_RESULTS_FILE)

    # Note: No optimization results to save (params loaded from file)
    opt_result = None

    # Optionally save a compact optimization summary
    if opt_result is not None:
        opt_path = LIGHTGBM_ARTIFACTS_DIR / "optimization_results.json"
        save_json_pretty(
            {
                "best_params": opt_result.best_params,
                "best_score": opt_result.best_score,
                "n_trials": opt_result.n_trials,
                "n_completed": opt_result.n_completed,
            },
            opt_path,
        )

    logger.info("Training complete. Model saved to %s", LIGHTGBM_MODEL_FILE)
    logger.info("Training metrics saved to %s", LIGHTGBM_TRAINING_RESULTS_FILE)


if __name__ == "__main__":
    main()
