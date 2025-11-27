"""Evaluate a trained Ridge Classifier on the held-out test split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).parent.parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config_logging import get_logger, setup_logging
from src.evaluation import evaluate_model, save_evaluation_results
from src.model.econometrie.ridge.main import load_data
from src.model.econometrie.ridge_classifier.ridge_classifier import RidgeClassifierModel
from src.path import (
    RIDGE_CLASSIFIER_ARTIFACTS_DIR,
    RIDGE_CLASSIFIER_MODEL_FILE,
    RIDGE_CLASSIFIER_TEST_EVAL_FILE,
)

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Ridge Classifier on test split")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=RIDGE_CLASSIFIER_MODEL_FILE,
        help=f"Path to trained model (default: {RIDGE_CLASSIFIER_MODEL_FILE})",
    )
    return parser.parse_args()


def main() -> None:
    """Load trained Ridge Classifier and evaluate on test split only."""
    args = parse_args()
    setup_logging()

    logger.info("=" * 60)
    logger.info("RIDGE CLASSIFIER EVALUATION (test split only)")
    logger.info("=" * 60)

    model_path = args.model_path
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. "
            "Run the training script first."
        )

    # Load data; keep train split untouched
    _, X_test, _, y_test = load_data()

    # Load model and evaluate
    model = RidgeClassifierModel.load(model_path)
    eval_result = evaluate_model(model, X_test, y_test, verbose=True)

    # Save evaluation
    RIDGE_CLASSIFIER_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    save_evaluation_results(eval_result, RIDGE_CLASSIFIER_TEST_EVAL_FILE)

    logger.info("Evaluation complete on %d samples", eval_result.n_samples)
    for metric, value in eval_result.metrics.items():
        logger.info("  %s: %.6f", metric, value)
    logger.info("Saved evaluation results to %s", RIDGE_CLASSIFIER_TEST_EVAL_FILE)


if __name__ == "__main__":
    main()
