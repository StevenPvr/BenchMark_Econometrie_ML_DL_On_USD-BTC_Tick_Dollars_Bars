"""Remove highly correlated features from clear datasets and save as final.

This script:
1. Loads dataset_features_clear (tree_based as reference)
2. Computes Spearman correlation on train data
3. Removes features with |correlation| > threshold
4. Plots correlation matrix AFTER cleaning
5. Applies to all datasets (tree_based, linear, lstm)
6. Saves to *_final.parquet files (ready for analyse_features/)

Pipeline: clear_features/ -> remove_correlated_features.py -> analyse_features/

Usage:
    python remove_correlated_features.py
    python remove_correlated_features.py --threshold 0.8
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path for direct execution.
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from typing import cast

import argparse

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from src.config_logging import get_logger, setup_logging
from src.path import (
    DATASET_FEATURES_CLEAR_PARQUET,
    DATASET_FEATURES_LINEAR_CLEAR_PARQUET,
    DATASET_FEATURES_LSTM_CLEAR_PARQUET,
    DATASET_FEATURES_FINAL_PARQUET,
    DATASET_FEATURES_LINEAR_FINAL_PARQUET,
    DATASET_FEATURES_LSTM_FINAL_PARQUET,
    DATA_DIR,
)

logger = get_logger(__name__)

# Input datasets (from clear_features/)
INPUT_DATASETS = {
    "tree_based": DATASET_FEATURES_CLEAR_PARQUET,
    "linear": DATASET_FEATURES_LINEAR_CLEAR_PARQUET,
    "lstm": DATASET_FEATURES_LSTM_CLEAR_PARQUET,
}

# Output datasets (final, ready for analyse_features/)
OUTPUT_DATASETS = {
    "tree_based": DATASET_FEATURES_FINAL_PARQUET,
    "linear": DATASET_FEATURES_LINEAR_FINAL_PARQUET,
    "lstm": DATASET_FEATURES_LSTM_FINAL_PARQUET,
}

# Output directory for plots
PLOTS_DIR = DATA_DIR / "analyse_features" / "plots" / "png"

# Columns to exclude from correlation analysis
META_COLUMNS = [
    "bar_id",
    "datetime_close",
    "split",
    "bar_id_lag1",
    "bar_id_lag5",
    "bar_id_lag10",
    "bar_id_lag15",
    "bar_id_lag25",
    "bar_id_lag50",
]
TARGET_COLUMN = "log_return"

# Default correlation threshold
DEFAULT_THRESHOLD = 0.7

# Features to manually remove (identified via SHAP/analysis)
MANUAL_FEATURES_TO_REMOVE = [
    # PCA distribution long
    "pca_distribution_long_c2",
    "pca_distribution_long_c3",
    "pca_distribution_long_c4",
    "pca_distribution_long_c5",
    "pca_distribution_long_c6",
    "pca_distribution_long_c7",
    # PCA distribution medium
    "pca_distribution_medium_c2",
    "pca_distribution_medium_c3",
    "pca_distribution_medium_c4",
    # PCA distribution short
    "pca_distribution_short_c4",
    "pca_distribution_short_c5",
    "pca_distribution_short_c6",
    "pca_distribution_short_c7",
    # PCA drawdown short
    "pca_drawdown_short_c0",
    "pca_drawdown_short_c1",
    # PCA jumps medium
    "pca_jumps_medium_c1",
    "pca_jumps_medium_c2",
    "pca_jumps_medium_c4",
    "pca_jumps_medium_c5",
    # PCA moving averages
    "pca_moving_averages_long_c4",
    "pca_moving_averages_medium_c2",
    "pca_moving_averages_short_c3",
    "pca_moving_averages_short_c5",
    # PCA other
    "pca_other_long_c2",
    "pca_other_long_c4",
    "pca_other_long_c5",
    "pca_other_medium_c4",
    "pca_other_medium_c5",
    "pca_other_short_c7",
    # PCA returns long
    "pca_returns_long_c8",
    # PCA temporal long
    "pca_temporal_long_c4",
    "pca_temporal_long_c6",
    "pca_temporal_long_c8",
    "pca_temporal_long_c14",
    "pca_temporal_long_c15",
    # PCA temporal medium
    "pca_temporal_medium_c10",
    # PCA temporal short
    "pca_temporal_short_c5",
    "pca_temporal_short_c12",
    "pca_temporal_short_c13",
    # PCA volume medium
    "pca_volume_medium_c6",
    "pca_volume_medium_c8",
    "pca_volume_medium_c9",
    "pca_volume_medium_c11",
    # PCA volume short
    "pca_volume_short_c5",
    "pca_volume_short_c6",
    "pca_volume_short_c8",
    "pca_volume_short_c11",
    "pca_volume_short_c12",
    # PCA jumps long (high VIF)
    "pca_jumps_long_c6",
]


def compute_spearman_matrix(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Compute Spearman correlation matrix."""
    logger.info("Computing Spearman correlation for %d features...", len(feature_cols))

    X = cast(np.ndarray, df[feature_cols].values)
    X = np.nan_to_num(X, nan=0.0)

    corr_matrix, _ = stats.spearmanr(X, nan_policy="omit")

    if len(feature_cols) == 1:
        corr_matrix = np.array([[1.0]])

    return pd.DataFrame(corr_matrix, index=pd.Index(feature_cols), columns=pd.Index(feature_cols))


def find_features_to_drop(corr_matrix: pd.DataFrame, threshold: float) -> list[str]:
    """Find features to drop based on correlation threshold.

    Removes features with correlation > threshold OR < -threshold.
    For each highly correlated pair, drops the one with higher average correlation.
    """
    features = corr_matrix.columns.tolist()
    n_features = len(features)
    to_drop = set()

    for i in range(n_features):
        if features[i] in to_drop:
            continue

        for j in range(i + 1, n_features):
            if features[j] in to_drop:
                continue

            corr_val = corr_matrix.iloc[i, j]

            # Check if correlation > threshold OR < -threshold
            if corr_val > threshold or corr_val < -threshold:
                # Calculate mean absolute correlation for each feature
                mean_corr_i = corr_matrix.iloc[i, :].drop(features[i]).abs().mean()
                mean_corr_j = corr_matrix.iloc[j, :].drop(features[j]).abs().mean()

                if mean_corr_i > mean_corr_j:
                    to_drop.add(features[i])
                else:
                    to_drop.add(features[j])

    return list(to_drop)


def plot_correlation_matrix(corr_matrix: pd.DataFrame, title: str, filename: str) -> None:
    """Plot and save correlation heatmap."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    n = len(corr_matrix)
    figsize = (max(12, n * 0.15), max(10, n * 0.15))

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        corr_matrix,
        annot=n <= 30,
        fmt=".2f" if n <= 30 else "",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )

    ax.set_title(title, fontsize=14, fontweight="bold")

    if n > 20:
        plt.xticks(rotation=90, fontsize=6)
        plt.yticks(rotation=0, fontsize=6)
    else:
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(rotation=0, fontsize=8)

    plt.tight_layout()

    output_path = PLOTS_DIR / f"{filename}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved: %s", output_path)


def main(threshold: float = DEFAULT_THRESHOLD) -> None:
    """Run correlation-based feature removal."""
    setup_logging()

    logger.info("=" * 70)
    logger.info("REMOVE CORRELATED FEATURES")
    logger.info("=" * 70)
    logger.info("Threshold: corr > %.2f OR corr < -%.2f", threshold, threshold)

    # Load reference dataset (from clear features)
    ref_path = INPUT_DATASETS["tree_based"]
    logger.info("\nLoading: %s", ref_path)

    df = pd.read_parquet(ref_path)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    # Filter to train
    if "split" in df.columns:
        df_train = df[df["split"] == "train"].copy()
        logger.info("Train split: %d rows", len(df_train))
    else:
        df_train = df.copy()

    # Get feature columns
    all_cols = df.columns.tolist()
    feature_cols = [c for c in all_cols if c not in META_COLUMNS and c != TARGET_COLUMN]
    logger.info("Feature columns: %d", len(feature_cols))

    # Compute correlation BEFORE
    corr_before = cast(pd.DataFrame, compute_spearman_matrix(cast(pd.DataFrame, df_train), feature_cols))

    high_corr_before = (((corr_before > threshold) | (corr_before < -threshold)) & (corr_before.abs() < 1.0)).sum().sum() // 2
    logger.info("High correlation pairs BEFORE (>%.2f or <-%.2f): %d", threshold, threshold, high_corr_before)

    # Find features to drop based on correlation
    features_to_drop_corr = find_features_to_drop(corr_before, threshold)
    logger.info("\nFeatures to drop (correlation): %d", len(features_to_drop_corr))

    # Add manual features to drop (from SHAP analysis)
    manual_to_drop = [f for f in MANUAL_FEATURES_TO_REMOVE if f in feature_cols]
    logger.info("Features to drop (manual/SHAP): %d", len(manual_to_drop))
    for f in manual_to_drop:
        logger.info("  - %s", f)

    # Combine both lists
    features_to_drop = list(set(features_to_drop_corr) | set(manual_to_drop))
    logger.info("\nTotal features to drop: %d", len(features_to_drop))

    if features_to_drop_corr:
        logger.info("From correlation:")
        for f in features_to_drop_corr[:10]:
            logger.info("  - %s", f)
        if len(features_to_drop_corr) > 10:
            logger.info("  ... and %d more", len(features_to_drop_corr) - 10)

    # Features to keep
    features_to_keep = [f for f in feature_cols if f not in features_to_drop]
    logger.info("Features to keep: %d", len(features_to_keep))

    # Compute correlation AFTER
    corr_after = cast(pd.DataFrame, compute_spearman_matrix(cast(pd.DataFrame, df_train), features_to_keep))

    high_corr_after = (((corr_after > threshold) | (corr_after < -threshold)) & (corr_after.abs() < 1.0)).sum().sum() // 2
    logger.info("High correlation pairs AFTER: %d", high_corr_after)

    # Plot AFTER
    plot_correlation_matrix(
        corr_after,
        f"Spearman Correlation AFTER Selection ({len(features_to_keep)} features)",
        "correlation_after_selection",
    )

    # Apply to all datasets
    logger.info("\n" + "=" * 70)
    logger.info("APPLYING TO ALL DATASETS")
    logger.info("=" * 70)

    meta_cols_present = [c for c in META_COLUMNS if c in all_cols]
    cols_to_keep = meta_cols_present + [TARGET_COLUMN] + features_to_keep

    for name in INPUT_DATASETS:
        input_path = INPUT_DATASETS[name]
        output_path = OUTPUT_DATASETS[name]
        logger.info("\n%s:", name)
        logger.info("  Input:  %s", input_path)
        logger.info("  Output: %s", output_path)

        if not input_path.exists():
            logger.warning("  Input not found, skipping")
            continue

        df_dataset = pd.read_parquet(input_path)
        original_cols = len(df_dataset.columns)

        cols_in_dataset = [c for c in cols_to_keep if c in df_dataset.columns]
        df_dataset = df_dataset[cols_in_dataset]

        df_dataset.to_parquet(output_path, index=False)

        logger.info("  %d -> %d columns (saved to *_final)", original_cols, len(df_dataset.columns))

    logger.info("\n" + "=" * 70)
    logger.info("DONE")
    logger.info("=" * 70)
    logger.info("Removed: %d features", len(features_to_drop))
    logger.info("Kept: %d features", len(features_to_keep))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove highly correlated features")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Correlation threshold (default: {DEFAULT_THRESHOLD})",
    )
    args = parser.parse_args()

    main(threshold=args.threshold)
