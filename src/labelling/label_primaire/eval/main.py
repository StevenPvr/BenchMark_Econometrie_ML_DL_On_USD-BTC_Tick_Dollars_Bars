"""CLI entry point for primary model evaluation."""

from src.config_logging import get_logger
from src.labelling.label_primaire.eval.logic import (
    ClassificationMetrics,
    EvaluationResult,
    compute_metrics,
    evaluate_model,
    get_features_path,
    get_labeled_features_path,
    get_trained_models,
    load_data,
    load_model,
    print_comparison,
    print_confusion_matrix,
    print_metrics,
    print_results,
)

__all__ = [
    "ClassificationMetrics",
    "EvaluationResult",
    "compute_metrics",
    "evaluate_model",
    "get_features_path",
    "get_labeled_features_path",
    "get_trained_models",
    "load_data",
    "load_model",
    "print_comparison",
    "print_confusion_matrix",
    "print_metrics",
    "print_results",
    "main",
]

logger = get_logger(__name__)


def _select_model_interactive() -> str | None:
    """Prompt user to select a trained model for evaluation."""
    models = get_trained_models()
    if not models:
        logger.info("No trained models found.")
        return None

    print("Available trained models:")
    for i, model in enumerate(models, 1):
        print(f"{i} - {model}")

    while True:
        try:
            choice = input("Select model number (or 'q' to quit): ").strip()
            if choice.lower() == 'q':
                return None
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                return models[idx]
            else:
                print(f"Please enter a number between 1 and {len(models)}")
        except ValueError:
            print("Please enter a valid number")


def main() -> None:
    """CLI entry point for model evaluation."""
    model = _select_model_interactive()
    if model is None:
        logger.info("No model selected, exiting.")
        return

    logger.info("Evaluating model: %s", model)
    try:
        result = evaluate_model(model)
        print_results(result)
        logger.info("Evaluation completed successfully")
    except Exception as e:
        logger.error("Evaluation failed: %s", e)


if __name__ == "__main__":  # pragma: no cover
    main()
