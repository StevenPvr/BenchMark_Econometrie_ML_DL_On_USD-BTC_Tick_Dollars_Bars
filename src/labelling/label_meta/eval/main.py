"""CLI entry point for meta-label model evaluation."""

from src.config_logging import get_logger
from src.labelling.label_meta.eval.logic import (
    CombinedEvaluationResult,
    TradingMetrics,
    compute_trading_metrics,
    evaluate_combined_performance,
    get_available_trained_meta_models,
    load_meta_model,
    load_meta_training_results,
)
from src.path import LABEL_META_TRAIN_DIR

logger = get_logger(__name__)

__all__ = [
    "CombinedEvaluationResult",
    "TradingMetrics",
    "compute_trading_metrics",
    "evaluate_combined_performance",
    "get_available_trained_meta_models",
    "load_meta_model",
    "load_meta_training_results",
    "main",
    "print_evaluation_results",
    "select_trained_meta_model",
]


def print_evaluation_results(result: CombinedEvaluationResult) -> None:
    """Print evaluation results in a readable format."""
    print(f"\n{'='*60}")
    print(f"EVALUATION: {result.primary_model_name} + {result.meta_model_name}")
    print(f"{'='*60}")
    print(f"\nSamples evaluated: {result.n_test_samples}")
    print(f"\nMETA MODEL METRICS:")
    print(f"  Accuracy:  {result.meta_accuracy:.4f}")
    print(f"  Precision: {result.meta_precision:.4f}")
    print(f"  Recall:    {result.meta_recall:.4f}")
    print(f"  F1 Score:  {result.meta_f1:.4f}")
    print(f"\nTRADING METRICS (Primary Only):")
    pm = result.primary_only_metrics
    print(f"  Trades:     {pm.get('n_trades', 0)}")
    print(f"  Win Rate:   {pm.get('win_rate', 0):.4f}")
    print(f"  Total Ret:  {pm.get('total_return', 0):.4f}")
    print(f"  Sharpe:     {pm.get('sharpe_ratio', 'N/A')}")
    print(f"\nTRADING METRICS (Combined with Meta):")
    cm = result.combined_metrics
    print(f"  Trades:     {cm.get('n_trades', 0)}")
    print(f"  Win Rate:   {cm.get('win_rate', 0):.4f}")
    print(f"  Total Ret:  {cm.get('total_return', 0):.4f}")
    print(f"  Sharpe:     {cm.get('sharpe_ratio', 'N/A')}")
    print(f"\nIMPROVEMENT:")
    print(f"  Trades filtered: {result.trades_filtered_pct:.2f}%")
    print(f"  Win rate change: {result.win_rate_improvement:+.4f}")
    print(f"{'='*60}\n")


def select_trained_meta_model() -> tuple[str, str] | None:
    """Prompt user to select a trained meta model."""
    available = get_available_trained_meta_models()
    if not available:
        logger.info("No trained meta models found.")
        return None
    logger.info("Select trained meta model:")
    for idx, (primary, meta) in enumerate(available, start=1):
        logger.info("%s - %s + %s", idx, primary, meta)
    choice = input("Enter choice (number): ").strip()
    if choice.isdigit():
        idx = int(choice) - 1
        return available[idx] if 0 <= idx < len(available) else None
    return None


def main() -> None:
    """CLI entry point for meta-label model evaluation."""
    selection = select_trained_meta_model()
    if selection is None:
        logger.info("No meta model selected, exiting.")
        return

    primary_model, meta_model = selection
    logger.info("Evaluating %s + %s...", primary_model, meta_model)

    try:
        result = evaluate_combined_performance(primary_model, meta_model)
        print_evaluation_results(result)

        save_input = input("Save results to file? [Y/n]: ").strip().lower()
        if save_input != "n":
            output_path = LABEL_META_TRAIN_DIR / f"{primary_model}_{meta_model}" / "evaluation_results.json"
            result.save(output_path)
            logger.info("Results saved to %s", output_path)

    except FileNotFoundError as e:
        logger.error("Error: %s", e)
    except Exception as e:
        logger.error("Evaluation failed: %s", e)
        raise


if __name__ == "__main__":  # pragma: no cover
    main()
