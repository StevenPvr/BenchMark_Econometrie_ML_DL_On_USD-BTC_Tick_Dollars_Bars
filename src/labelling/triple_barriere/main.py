"""CLI entry point for triple-barrier relabeling with Optuna optimization."""

from __future__ import annotations


from pathlib import Path
import sys

# Add project root to Python path for direct execution.
# This must be done before importing src modules.
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config_logging import get_logger
from src.labelling.triple_barriere.relabel_datasets import (
    DATASET_FRACTION,
    N_TRIALS,
    SEARCH_SPACE,
    SOURCE_DATASET,
    TripleBarrierOptimizer,
    VOL_SPAN,
    apply_labels_to_datasets,
    get_feature_columns,
    prepare_data,
)

logger = get_logger(__name__)


def main(n_trials: int = N_TRIALS) -> None:
    """Run Optuna optimization for triple-barrier labels and apply best labels.

    Args:
        n_trials: Number of Optuna trials to run.
    """
    logger.info("Loading source dataset: %s (train split only)", SOURCE_DATASET)
    features_df, close, volatility = prepare_data(
        vol_span=VOL_SPAN,
        fraction=DATASET_FRACTION,
        use_train_only=True,  # Only optimize on train to avoid data leakage
    )

    logger.info(
        "Dataset loaded: %d samples, %d features",
        len(features_df),
        len(get_feature_columns(features_df)),
    )

    optimizer = TripleBarrierOptimizer(
        search_space=SEARCH_SPACE,
        n_trials=n_trials,
        vol_span=VOL_SPAN,
    )
    optimizer.fit(features_df, close, volatility)

    if optimizer.best_params_ is None or optimizer.best_metrics_ is None:
        logger.warning("No valid configuration found. Datasets not modified.")
        return

    apply_labels_to_datasets(
        best_params=optimizer.best_params_,
        best_metrics=optimizer.best_metrics_,
    )

    results_df = optimizer.get_results_dataframe()
    results_path = Path(__file__).parent / "optuna_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info("Results saved to: %s", results_path)


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Optimize triple-barrier labels with Optuna")
    parser.add_argument(
        "--n-trials",
        type=int,
        default=N_TRIALS,
        help=f"Number of Optuna trials (default: {N_TRIALS})",
    )
    args = parser.parse_args()

    main(n_trials=args.n_trials)
