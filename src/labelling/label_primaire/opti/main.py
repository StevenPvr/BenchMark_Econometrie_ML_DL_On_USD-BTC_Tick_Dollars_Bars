"""CLI entry point for primary labeling optimization."""

from __future__ import annotations

from pathlib import Path
import sys

# Add project root to Python path for direct execution.
# This must be done before importing src modules.
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


from src.config_logging import get_logger
from src.labelling.label_primaire.opti.logic import (
    OptimizationConfig,
    OptimizationResult,
    WalkForwardCV,
    create_objective,
    optimize_model,
    print_final_summary,
    run_parallel,
    run_sequential,
    select_models_interactive,
)

__all__ = [
    "OptimizationConfig",
    "OptimizationResult",
    "WalkForwardCV",
    "create_objective",
    "optimize_model",
    "print_final_summary",
    "run_parallel",
    "run_sequential",
    "select_models_interactive",
    "main",
]

logger = get_logger(__name__)


def main() -> None:
    """Entry point for CLI usage."""
    models = select_models_interactive()
    if not models:
        logger.info("No models selected, exiting.")
        return
    trials_input = input("Number of trials per model (default 10): ").strip()
    trials = int(trials_input) if trials_input else 10
    splits_input = input("Number of CV splits (default 5): ").strip()
    n_splits = int(splits_input) if splits_input else 5
    use_parallel = input("Run in parallel? (y/n) [n]: ").strip().lower() == "y"
    trials_per_model = {name: trials for name in models}
    results = run_parallel(models, trials_per_model, n_splits) if use_parallel else run_sequential(models, trials_per_model, n_splits)
    print_final_summary(results)


if __name__ == "__main__":  # pragma: no cover
    main()
