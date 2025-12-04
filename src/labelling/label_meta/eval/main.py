"""CLI entry point for meta-label model evaluation."""

from pathlib import Path
import sys

# Add project root to Python path for direct execution.
# This must be done before importing src modules.
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config_logging import get_logger
from src.labelling.label_meta.eval.logic import (
    INITIAL_CAPITAL,
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


def _fmt_eur(val: float | None) -> str:
    """Format a value in euros."""
    return f"{val:,.2f}€" if val is not None else "N/A"


def _fmt_pct(val: float | None) -> str:
    """Format a value as percentage."""
    return f"{val:.2f}%" if val is not None else "N/A"


def _fmt(val: float | None, decimals: int = 2) -> str:
    """Format a value or return N/A if None."""
    return f"{val:.{decimals}f}" if val is not None else "N/A"


def print_evaluation_results(result: CombinedEvaluationResult) -> None:
    """Print evaluation results in a readable format."""
    print(f"\n{'='*70}")
    print(f"EVALUATION: {result.primary_model_name} + {result.meta_model_name}")
    print(f"{'='*70}")
    print(f"\nSamples evaluated: {result.n_test_samples}")
    print(f"Initial capital: {INITIAL_CAPITAL:,.0f}€")

    print(f"\n{'─'*70}")
    print("META MODEL METRICS (predicting if primary is correct):")
    print(f"{'─'*70}")
    print(f"  Accuracy:  {result.meta_accuracy:.4f}")
    print(f"  Precision: {result.meta_precision:.4f}")
    print(f"  Recall:    {result.meta_recall:.4f}")
    print(f"  F1 Score:  {result.meta_f1:.4f}")

    pm = result.primary_only_metrics
    cm = result.combined_metrics

    print(f"\n{'─'*70}")
    print("P&L COMPARISON (with 1% transaction costs per trade)")
    print(f"{'─'*70}")
    print(f"{'Metric':<25} {'Primary Only':>20} {'Primary + Meta':>20}")
    print(f"{'─'*70}")

    # Trade stats
    print(f"{'Trades':<25} {pm.get('n_trades', 0):>20,} {cm.get('n_trades', 0):>20,}")
    # Classification win rate (correct predictions)
    print(f"{'Accuracy (classification)':<25} {_fmt_pct(pm.get('classification_win_rate', 0) * 100):>20} {_fmt_pct(cm.get('classification_win_rate', 0) * 100):>20}")
    # Trading win rate (net profitable trades after costs)
    print(f"{'Win Rate (net P&L > 0)':<25} {_fmt_pct(pm.get('win_rate', 0) * 100):>20} {_fmt_pct(cm.get('win_rate', 0) * 100):>20}")

    # P&L in euros
    print(f"\n{'─'*70}")
    print("P&L (in euros):")
    print(f"{'─'*70}")
    print(f"{'Gross P&L':<25} {_fmt_eur(pm.get('gross_pnl')):>20} {_fmt_eur(cm.get('gross_pnl')):>20}")
    print(f"{'Total Costs':<25} {_fmt_eur(pm.get('total_costs')):>20} {_fmt_eur(cm.get('total_costs')):>20}")
    print(f"{'Net P&L':<25} {_fmt_eur(pm.get('net_pnl')):>20} {_fmt_eur(cm.get('net_pnl')):>20}")
    print(f"{'Final Capital':<25} {_fmt_eur(pm.get('final_capital')):>20} {_fmt_eur(cm.get('final_capital')):>20}")

    # Returns as percentages
    print(f"\n{'─'*70}")
    print("Returns:")
    print(f"{'─'*70}")
    print(f"{'Gross Return':<25} {_fmt_pct(pm.get('gross_return_pct')):>20} {_fmt_pct(cm.get('gross_return_pct')):>20}")
    print(f"{'Net Return':<25} {_fmt_pct(pm.get('net_return_pct')):>20} {_fmt_pct(cm.get('net_return_pct')):>20}")

    # Risk metrics
    print(f"\n{'─'*70}")
    print("Risk Metrics:")
    print(f"{'─'*70}")
    print(f"{'Sharpe Ratio':<25} {_fmt(pm.get('sharpe_ratio')):>20} {_fmt(cm.get('sharpe_ratio')):>20}")
    print(f"{'Max Drawdown':<25} {_fmt_eur(pm.get('max_drawdown')):>20} {_fmt_eur(cm.get('max_drawdown')):>20}")
    print(f"{'Max Drawdown %':<25} {_fmt_pct(pm.get('max_drawdown_pct')):>20} {_fmt_pct(cm.get('max_drawdown_pct')):>20}")
    print(f"{'Profit Factor':<25} {_fmt(pm.get('profit_factor')):>20} {_fmt(cm.get('profit_factor')):>20}")

    # Win/loss analysis
    print(f"\n{'─'*70}")
    print("Trade Analysis:")
    print(f"{'─'*70}")
    print(f"{'Avg Win':<25} {_fmt_eur(pm.get('avg_win')):>20} {_fmt_eur(cm.get('avg_win')):>20}")
    print(f"{'Avg Loss':<25} {_fmt_eur(pm.get('avg_loss')):>20} {_fmt_eur(cm.get('avg_loss')):>20}")

    print(f"\n{'─'*70}")
    print("IMPROVEMENT SUMMARY:")
    print(f"{'─'*70}")
    print(f"  Trades filtered:      {result.trades_filtered_pct:.2f}%")
    print(f"  Accuracy change:      {result.win_rate_improvement * 100:+.2f}% (classification)")

    # Net P&L improvement
    net_pnl_pm = pm.get('net_pnl', 0) or 0
    net_pnl_cm = cm.get('net_pnl', 0) or 0
    net_improvement = net_pnl_cm - net_pnl_pm
    print(f"  Net P&L change:     {net_improvement:+,.2f}€")

    # Sharpe improvement
    sharpe_pm = pm.get('sharpe_ratio')
    sharpe_cm = cm.get('sharpe_ratio')
    if sharpe_pm is not None and sharpe_cm is not None:
        print(f"  Sharpe change:      {sharpe_cm - sharpe_pm:+.2f}")

    print(f"{'='*70}\n")


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
