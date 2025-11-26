"""Evaluate a trained LightGBM model on the held-out test split."""

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
from src.model.machine_learning.lightgbm.main import load_data
from src.model.machine_learning.lightgbm.lightgbm_model import LightGBMModel
from src.path import (
    LIGHTGBM_ARTIFACTS_DIR,
    LIGHTGBM_MODEL_FILE,
    LIGHTGBM_TEST_EVAL_FILE,
)

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate LightGBM on test split")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=LIGHTGBM_MODEL_FILE,
        help=f"Path to trained model (default: {LIGHTGBM_MODEL_FILE})",
    )
    return parser.parse_args()


def main() -> None:
    """Load trained model and evaluate on test split only."""
    args = parse_args()
    setup_logging()

    logger.info("=" * 60)
    logger.info("LIGHTGBM EVALUATION (test split only)")
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
    model = LightGBMModel.load(model_path)
    eval_result = evaluate_model(model, X_test, y_test, verbose=True)

    # Save evaluation
    LIGHTGBM_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    save_evaluation_results(eval_result, LIGHTGBM_TEST_EVAL_FILE)

    logger.info("Evaluation complete on %d samples", eval_result.n_samples)
    for metric, value in eval_result.metrics.items():
        logger.info("  %s: %.6f", metric, value)
    logger.info("Saved evaluation results to %s", LIGHTGBM_TEST_EVAL_FILE)


if __name__ == "__main__":
    main()
