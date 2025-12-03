"""Automated feature selection based on analysis results.

This module provides functions to select optimal features based on:
- Stationarity tests (ADF/KPSS)
- Multicollinearity (VIF)
- Target correlation/importance
- Granger causality
- Normality (optional)

The goal is to automate the removal of problematic features:
- Non-stationary features (can cause spurious regressions)
- Highly collinear features (VIF > threshold)
- Features with no relationship to target

Use cases:
- Create curated feature sets for modeling
- Reduce dimensionality while preserving information
- Ensure statistical assumptions are met
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from src.analyse_features.config import SAVE_BATCH_SIZE, TARGET_COLUMN
from src.config_logging import get_logger

logger = get_logger(__name__)


def get_recommended_features(
    stationarity_df: pd.DataFrame | None = None,
    vif_df: pd.DataFrame | None = None,
    target_metrics_df: pd.DataFrame | None = None,
    granger_df: pd.DataFrame | None = None,
    normality_df: pd.DataFrame | None = None,
    all_features: list[str] | None = None,
    max_vif: float = 10.0,
    min_target_corr: float = 0.01,
    require_stationary: bool = True,
    require_normal: bool = False,
    require_granger: bool = False,
) -> dict[str, Any]:
    """Select features based on multiple analysis criteria.

    Applies filters sequentially and tracks which features are removed at each step.

    Args:
        stationarity_df: DataFrame with stationarity results (needs 'stationarity_conclusion').
        vif_df: DataFrame with VIF scores (needs 'feature', 'vif').
        target_metrics_df: DataFrame with target metrics (needs 'feature', 'abs_spearman').
        granger_df: DataFrame with Granger results (needs 'feature', 'granger_significant').
        normality_df: DataFrame with normality results (needs 'feature', 'is_normal').
        all_features: Complete list of features (used if any df is None).
        max_vif: Maximum VIF threshold (features above are removed).
        min_target_corr: Minimum |Spearman| with target.
        require_stationary: If True, remove non-stationary features.
        require_normal: If True, remove non-normal features.
        require_granger: If True, remove features without Granger causality.

    Returns:
        Dictionary with:
            - 'selected': List of selected feature names
            - 'removed': Dict mapping reason -> removed features
            - 'stats': Summary statistics
    """
    logger.info("Running feature selection with criteria:")
    logger.info("  - max_vif: %.1f", max_vif)
    logger.info("  - min_target_corr: %.3f", min_target_corr)
    logger.info("  - require_stationary: %s", require_stationary)
    logger.info("  - require_normal: %s", require_normal)
    logger.info("  - require_granger: %s", require_granger)

    # Determine initial feature set
    if all_features is not None:
        current_features = set(all_features)
    elif stationarity_df is not None and "feature" in stationarity_df.columns:
        current_features = set(stationarity_df["feature"].tolist())
    elif vif_df is not None and "feature" in vif_df.columns:
        current_features = set(vif_df["feature"].tolist())
    elif target_metrics_df is not None and "feature" in target_metrics_df.columns:
        current_features = set(target_metrics_df["feature"].tolist())
    else:
        raise ValueError("No feature list provided. Pass all_features or at least one analysis DataFrame.")

    initial_count = len(current_features)
    logger.info("Starting with %d features", initial_count)

    removed: dict[str, list[str]] = {}

    # 1. Filter by stationarity
    if require_stationary and stationarity_df is not None and "stationarity_conclusion" in stationarity_df.columns:
        stationary_features = set(stationarity_df[
            stationarity_df["stationarity_conclusion"].isin(["stationary", "trend_stationary"])
        ]["feature"])

        removed_stationary = current_features - stationary_features
        if removed_stationary:
            removed["non_stationary"] = sorted(list(removed_stationary))
            logger.info("Removed %d non-stationary features", len(removed_stationary))

        current_features = current_features & stationary_features

    # 2. Filter by VIF
    if vif_df is not None and "vif" in vif_df.columns:
        low_vif_features = set(vif_df[vif_df["vif"] <= max_vif]["feature"])

        removed_vif = current_features - low_vif_features
        if removed_vif:
            removed["high_vif"] = sorted(list(removed_vif))
            logger.info("Removed %d high-VIF features (VIF > %.1f)", len(removed_vif), max_vif)

        current_features = current_features & low_vif_features

    # 3. Filter by target correlation
    if target_metrics_df is not None and "abs_spearman" in target_metrics_df.columns:
        relevant_features = set(target_metrics_df[
            target_metrics_df["abs_spearman"] >= min_target_corr
        ]["feature"])

        removed_corr = current_features - relevant_features
        if removed_corr:
            removed["low_target_corr"] = sorted(list(removed_corr))
            logger.info("Removed %d features with low target correlation (|rho| < %.3f)",
                       len(removed_corr), min_target_corr)

        current_features = current_features & relevant_features

    # 4. Filter by Granger causality (optional)
    if require_granger and granger_df is not None and "granger_significant" in granger_df.columns:
        granger_features = set(granger_df[
            granger_df["granger_significant"] == True  # noqa: E712
        ]["feature"])

        removed_granger = current_features - granger_features
        if removed_granger:
            removed["no_granger_causality"] = sorted(list(removed_granger))
            logger.info("Removed %d features without Granger causality", len(removed_granger))

        current_features = current_features & granger_features

    # 5. Filter by normality (optional)
    if require_normal and normality_df is not None and "is_normal" in normality_df.columns:
        normal_features = set(normality_df[
            normality_df["is_normal"] == True  # noqa: E712
        ]["feature"])

        removed_normal = current_features - normal_features
        if removed_normal:
            removed["non_normal"] = sorted(list(removed_normal))
            logger.info("Removed %d non-normal features", len(removed_normal))

        current_features = current_features & normal_features

    # Final selection
    selected = sorted(list(current_features))
    final_count = len(selected)

    stats = {
        "initial_count": initial_count,
        "final_count": final_count,
        "removed_count": initial_count - final_count,
        "retention_rate": final_count / initial_count if initial_count > 0 else 0,
    }

    logger.info("Feature selection complete: %d -> %d features (%.1f%% retained)",
               initial_count, final_count, stats["retention_rate"] * 100)

    return {
        "selected": selected,
        "removed": removed,
        "stats": stats,
    }


def rank_features(
    target_metrics_df: pd.DataFrame | None = None,
    granger_df: pd.DataFrame | None = None,
    stationarity_df: pd.DataFrame | None = None,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Rank features by importance using multiple metrics.

    Creates a combined ranking based on:
    - Target correlation (Spearman)
    - Mutual Information
    - Granger causality p-value

    Args:
        target_metrics_df: DataFrame with target metrics.
        granger_df: DataFrame with Granger results.
        stationarity_df: DataFrame with stationarity results.
        top_n: Return only top N features (None = all).

    Returns:
        DataFrame with feature rankings.
    """
    if target_metrics_df is None:
        raise ValueError("target_metrics_df is required for ranking")

    # Start with target metrics
    result = target_metrics_df[["feature"]].copy()

    # Add correlation rank
    if "abs_spearman" in target_metrics_df.columns:
        result = result.merge(
            target_metrics_df[["feature", "abs_spearman"]],
            on="feature",
            how="left",
        )
        result["rank_correlation"] = result["abs_spearman"].rank(ascending=False)

    # Add MI rank
    if "mutual_information" in target_metrics_df.columns:
        result = result.merge(
            target_metrics_df[["feature", "mutual_information"]],
            on="feature",
            how="left",
        )
        result["rank_mi"] = result["mutual_information"].rank(ascending=False)

    # Add Granger rank
    if granger_df is not None and "granger_pvalue" in granger_df.columns:
        result = result.merge(
            granger_df[["feature", "granger_pvalue", "granger_significant"]],
            on="feature",
            how="left",
        )
        # Lower p-value = better = higher rank
        result["rank_granger"] = result["granger_pvalue"].rank(ascending=True)

    # Add stationarity flag
    if stationarity_df is not None and "stationarity_conclusion" in stationarity_df.columns:
        result = result.merge(
            stationarity_df[["feature", "stationarity_conclusion"]],
            on="feature",
            how="left",
        )
        result["is_stationary"] = result["stationarity_conclusion"].isin(
            ["stationary", "trend_stationary"]
        )

    # Compute combined rank
    rank_columns = [c for c in result.columns if c.startswith("rank_")]
    if rank_columns:
        result = cast(pd.DataFrame, result)
        result["combined_rank"] = result[rank_columns].mean(axis=1)
        result = cast(pd.DataFrame, result.sort_values(by=["combined_rank"])).reset_index(drop=True)

    if top_n is not None:
        result = cast(pd.DataFrame, result.head(top_n))

    return cast(pd.DataFrame, result)


def _save_parquet_batch(df: pd.DataFrame, path: Path, batch_size: int = SAVE_BATCH_SIZE) -> None:
    """Save DataFrame to parquet by batch (memory efficient)."""
    n_rows = len(df)
    n_batches = (n_rows + batch_size - 1) // batch_size

    parquet_writer: pq.ParquetWriter | None = None

    for batch_num in range(n_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, n_rows)

        df_batch = df.iloc[start_idx:end_idx]
        table = pa.Table.from_pandas(df_batch, preserve_index=False)

        if parquet_writer is None:
            parquet_writer = pq.ParquetWriter(path, table.schema, compression="snappy")

        parquet_writer.write_table(table)

        del df_batch, table
        gc.collect()

    if parquet_writer is not None:
        parquet_writer.close()


def export_feature_subset(
    df: pd.DataFrame,
    selected_features: list[str],
    output_path: str | Path,
    include_target: bool = True,
    target_column: str = TARGET_COLUMN,
) -> None:
    """Export a DataFrame with only selected features.

    Args:
        df: Original DataFrame.
        selected_features: List of feature columns to keep.
        output_path: Path to save the result.
        include_target: Whether to include the target column.
        target_column: Name of target column.
    """
    columns_to_export = list(selected_features)

    if include_target and target_column in df.columns and target_column not in columns_to_export:
        columns_to_export.append(target_column)

    # Keep any non-feature columns that might be useful (split, timestamp, etc.)
    for col in ["split", "timestamp", "datetime", "date", "index"]:
        if col in df.columns and col not in columns_to_export:
            columns_to_export.insert(0, col)

    # Filter to existing columns
    available_columns = [c for c in columns_to_export if c in df.columns]
    missing = set(columns_to_export) - set(available_columns)
    if missing:
        logger.warning("Some columns not found in DataFrame: %s", missing)

    # Export
    output_df = cast(pd.DataFrame, df[available_columns].copy())
    output_path = Path(output_path)

    if output_path.suffix == ".parquet":
        _save_parquet_batch(output_df, output_path)
    elif output_path.suffix == ".csv":
        output_df.to_csv(output_path, index=False)
    else:
        # Default to parquet
        _save_parquet_batch(output_df, output_path.with_suffix(".parquet"))

    logger.info("Exported %d columns to %s", len(available_columns), output_path)


if __name__ == "__main__":
    from src.config_logging import setup_logging

    setup_logging()

    # Example usage
    logger.info("Feature selection module loaded successfully")
    logger.info("Available functions:")
    logger.info("  - get_recommended_features(): Select features by criteria")
    logger.info("  - rank_features(): Rank features by importance")
    logger.info("  - export_feature_subset(): Export selected features to file")
