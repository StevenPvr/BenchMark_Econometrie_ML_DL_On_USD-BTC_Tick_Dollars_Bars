"""CLI entry point for meta-label model training."""


from pathlib import Path
import sys

# Add project root to Python path for direct execution.
# This must be done before importing src modules.
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config_logging import get_logger
from src.labelling.label_meta.train.logic import (
    MetaEvaluationMetrics,
    MetaTrainingConfig,
    MetaTrainingResult,
    get_available_meta_optimizations,
    get_yes_no_input,
    load_optimized_params,
    select_meta_model,
    select_primary_model,
    train_meta_model,
)

logger = get_logger(__name__)

__all__ = [
    "MetaEvaluationMetrics",
    "MetaTrainingConfig",
    "MetaTrainingResult",
    "get_available_meta_optimizations",
    "get_yes_no_input",
    "load_optimized_params",
    "main",
    "select_meta_model",
    "select_primary_model",
    "train_meta_model",
]


def main() -> None:
    """CLI entry point for meta model training."""
    primary_model = select_primary_model()
    if primary_model is None:
        logger.info("No primary model selected, exiting.")
        return

    meta_model = select_meta_model(primary_model)
    if meta_model is None:
        logger.info("No meta model selected, exiting.")
        return

    try:
        params = load_optimized_params(primary_model, meta_model)
        logger.info(
            "Best %s: %.4f for primary=%s, meta=%s",
            params.get("metric", "score"),
            params.get("best_score", 0.0),
            primary_model,
            meta_model,
        )
    except FileNotFoundError:
        logger.error("Optimization file not found for %s_%s", primary_model, meta_model)
        return

    if not get_yes_no_input("Train this meta model?", default=True):
        logger.info("Training cancelled.")
        return

    result = train_meta_model(primary_model, meta_model)
    logger.info(
        "Trained meta model %s_%s on %s samples",
        primary_model,
        meta_model,
        result.train_samples,
    )
    logger.info("Train metrics: %s", result.train_metrics)


if __name__ == "__main__":  # pragma: no cover
    main()
