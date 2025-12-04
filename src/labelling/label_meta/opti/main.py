"""CLI entry point for meta-labeling optimization."""


from pathlib import Path
import sys

# Add project root to Python path for direct execution.
# This must be done before importing src modules.
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config_logging import get_logger
from src.labelling.label_meta.opti.logic import (
    MODEL_REGISTRY,
    WalkForwardCV,
    optimize_model,
)
from src.labelling.label_meta.utils import (
    MetaOptimizationConfig,
    MetaOptimizationResult,
    get_available_primary_models,
)

logger = get_logger(__name__)

__all__ = [
    "MetaOptimizationConfig",
    "MetaOptimizationResult",
    "WalkForwardCV",
    "main",
    "optimize_model",
    "select_meta_model_interactive",
    "select_primary_model_interactive",
]


def select_primary_model_interactive() -> str | None:
    """Prompt user to select a primary model."""
    available = get_available_primary_models()
    if not available:
        logger.info("No trained primary models found.")
        return None
    logger.info("Select primary model:")
    for idx, name in enumerate(available, start=1):
        logger.info("%s - %s", idx, name)
    choice = input("Enter choice (number or name): ").strip()
    if choice.isdigit():
        idx = int(choice) - 1
        return available[idx] if 0 <= idx < len(available) else None
    return choice if choice in available else None


def select_meta_model_interactive() -> str | None:
    """Prompt user to select a meta model type."""
    model_names = list(MODEL_REGISTRY.keys())
    logger.info("Select meta model:")
    for idx, name in enumerate(model_names, start=1):
        logger.info("%s - %s", idx, name)
    choice = input("Enter choice (number or name): ").strip()
    if choice.isdigit():
        idx = int(choice) - 1
        return model_names[idx] if 0 <= idx < len(model_names) else None
    return choice if choice in model_names else None


def main() -> None:
    """CLI entry point for meta-label optimization."""
    primary_model = select_primary_model_interactive()
    if primary_model is None:
        logger.info("No primary model selected, exiting.")
        return

    meta_model = select_meta_model_interactive()
    if meta_model is None:
        logger.info("No meta model selected, exiting.")
        return

    trials_input = input("Number of trials (default 50): ").strip()
    n_trials = int(trials_input) if trials_input else 50

    splits_input = input("Number of CV splits (default 5): ").strip()
    n_splits = int(splits_input) if splits_input else 5

    config = MetaOptimizationConfig(
        primary_model_name=primary_model,
        meta_model_name=meta_model,
        n_trials=n_trials,
        n_splits=n_splits,
    )

    logger.info(
        "Starting optimization: primary=%s, meta=%s, trials=%s, splits=%s",
        primary_model,
        meta_model,
        n_trials,
        n_splits,
    )

    result = optimize_model(config)
    logger.info(
        "Optimization complete: best MCC=%.4f after %s trials",
        result.best_score,
        result.n_trials,
    )
    logger.info("Best params: %s", result.best_params)


if __name__ == "__main__":  # pragma: no cover
    main()
