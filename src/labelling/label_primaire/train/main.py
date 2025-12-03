"""CLI entry point for primary model training with OOF predictions."""

from src.config_logging import get_logger
from src.labelling.label_primaire.train.logic import (
    PrimaryEvaluationMetrics,
    TrainingConfig,
    TrainingResult,
    WalkForwardKFold,
    evaluate_model,
    get_available_optimized_models,
    get_yes_no_input,
    load_optimized_params,
    select_model,
    train_model,
)

logger = get_logger(__name__)

__all__ = [
    "PrimaryEvaluationMetrics",
    "TrainingConfig",
    "TrainingResult",
    "WalkForwardKFold",
    "evaluate_model",
    "get_available_optimized_models",
    "get_yes_no_input",
    "load_optimized_params",
    "main",
    "select_model",
    "train_model",
]


def main() -> None:
    """CLI entry point for primary model training."""
    model = select_model()
    if model is None:
        logger.info("No model selected, exiting.")
        return

    try:
        params = load_optimized_params(model)
        logger.info(
            "Best %s: %.4f for model %s",
            params.get("metric", "score"),
            params.get("best_score", 0.0),
            model,
        )
    except FileNotFoundError:
        logger.error("Optimization file not found for %s", model)
        return

    if not get_yes_no_input("Train this model?", default=True):
        logger.info("Training cancelled.")
        return

    result = train_model(model)
    logger.info(
        "Trained primary model %s on %s samples",
        model,
        result.train_samples,
    )
    logger.info("Train metrics: %s", result.train_metrics)
    logger.info("OOF predictions saved to: %s", result.oof_predictions_path)


if __name__ == "__main__":  # pragma: no cover
    main()
