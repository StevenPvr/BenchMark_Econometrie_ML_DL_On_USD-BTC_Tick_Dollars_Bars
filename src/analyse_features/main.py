"""Main orchestrator for feature analysis pipeline.

This module coordinates all feature analysis components:
1. Correlation Analysis (Spearman, MI)
2. Multicollinearity Analysis (VIF, Condition Number)
3. Target Relationship Analysis
4. Feature Clustering (Hierarchical, t-SNE, UMAP)
5. Temporal Analysis (ACF/PACF, Rolling Correlations)

Usage:
    # Run all analyses
    python -m src.analyse_features.main

    # Run specific analysis
    python -m src.analyse_features.main --analysis correlation

    # Skip slow analyses
    python -m src.analyse_features.main --skip-umap
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path for direct execution.
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]

from src.analyse_features.config import (
    ANALYSE_FEATURES_DIR,
    DATASET_SAMPLE_FRACTION,
    INPUT_DATASETS,
    SUMMARY_JSON,
    TARGET_COLUMN,
    ensure_directories,
)
from src.analyse_features.clustering import run_clustering_analysis
from src.analyse_features.correlation import run_correlation_analysis
from src.analyse_features.multicollinearity import run_multicollinearity_analysis
from src.analyse_features.target_analysis import run_target_analysis
from src.analyse_features.temporal import run_temporal_analysis
from src.analyse_features.utils.json_utils import save_json
from src.analyse_features.utils.plotting import (
    plot_correlation_heatmap,
    plot_dendrogram,
    plot_embedding,
    plot_target_correlations,
    plot_vif_scores,
)
from src.config_logging import get_logger, setup_logging
from src.constants import DEFAULT_RANDOM_STATE

logger = get_logger(__name__)


def load_features(
    file_path: Path | None = None,
    use_train_only: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Load feature dataset and extract feature columns.

    Args:
        file_path: Path to features parquet (default: INPUT_DATASETS["lstm"]).
        use_train_only: Whether to use only training data.

    Returns:
        Tuple of (DataFrame, list of feature column names).
    """
    file_path = file_path or INPUT_DATASETS["lstm"]

    logger.info("Loading features from %s", file_path)
    df = pd.read_parquet(file_path)

    initial_rows = len(df)

    if use_train_only and "split" in df.columns:
        df = df[df["split"] == "train"].copy()
        df = df.drop(columns=["split"])
        logger.info("Filtered to train split: %d -> %d rows", initial_rows, len(df))

    if DATASET_SAMPLE_FRACTION < 1.0:
        rows_before_sample = len(df)
        df = df.sample(
            frac=DATASET_SAMPLE_FRACTION,
            random_state=DEFAULT_RANDOM_STATE,
        ).reset_index(drop=True)
        logger.info(
            "Sampled %.1f%% of dataset: %d -> %d rows",
            DATASET_SAMPLE_FRACTION * 100,
            rows_before_sample,
            len(df),
        )

    exclude_cols = {TARGET_COLUMN, "split", "index", "timestamp", "datetime", "date"}
    numeric_cols = df.select_dtypes(include=["number"]).columns
    feature_columns: list[str] = [str(col) for col in numeric_cols if col not in exclude_cols]

    logger.info("Found %d feature columns", len(feature_columns))

    return cast(pd.DataFrame, df), feature_columns


def _save_full_summary(all_results: dict[str, Any], total_time: float) -> None:
    """Save complete analysis summary to JSON."""
    summary: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "total_time_seconds": total_time,
        "analyses_completed": list(all_results.keys()),
    }

    if "correlation" in all_results and "spearman" in all_results["correlation"]:
        spearman = all_results["correlation"]["spearman"]
        high_corr = ((spearman.abs() > 0.7) & (spearman.abs() < 1.0)).sum().sum() // 2
        summary["correlation"] = {
            "n_features": len(spearman),
            "high_correlation_pairs": int(high_corr),
        }

    if "multicollinearity" in all_results and "vif" in all_results["multicollinearity"]:
        vif_df = all_results["multicollinearity"]["vif"]
        summary["multicollinearity"] = {
            "features_vif_over_10": int((vif_df["vif"] > 10).sum()),
            "features_vif_over_100": int((vif_df["vif"] > 100).sum()),
        }

    if "target" in all_results and "summary" in all_results["target"]:
        summary["target"] = all_results["target"]["summary"]

    if "clustering" in all_results and "clusters" in all_results["clustering"]:
        clusters_df = all_results["clustering"]["clusters"]
        summary["clustering"] = {"n_clusters": int(clusters_df["cluster"].nunique())}

    save_json(summary, SUMMARY_JSON)


def _generate_clustering_plots(cluster_results: dict[str, Any], feature_columns: list[str]) -> None:
    """Generate clustering visualizations."""
    if "hierarchical" in cluster_results:
        plot_dendrogram(
            cluster_results["hierarchical"]["linkage_matrix"],
            feature_columns,
            title="Feature Hierarchical Clustering",
            filename="feature_dendrogram",
        )

    if "tsne" in cluster_results and "clusters" in cluster_results:
        tsne_df = cluster_results["tsne"].merge(cluster_results["clusters"], on="feature")
        plot_embedding(
            tsne_df,
            x_col="tsne_1",
            y_col="tsne_2",
            label_col="feature",
            color_col="cluster",
            title="t-SNE Feature Embedding",
            filename="tsne_embedding",
        )

    if "umap" in cluster_results and "clusters" in cluster_results:
        umap_df = cluster_results["umap"].merge(cluster_results["clusters"], on="feature")
        if not umap_df["umap_1"].isna().all():
            plot_embedding(
                umap_df,
                x_col="umap_1",
                y_col="umap_2",
                label_col="feature",
                color_col="cluster",
                title="UMAP Feature Embedding",
                filename="umap_embedding",
            )


def run_all_analyses(
    df: pd.DataFrame,
    feature_columns: list[str],
    compute_umap: bool = True,
    generate_plots: bool = True,
) -> dict[str, Any]:
    """Run complete feature analysis pipeline.

    Args:
        df: DataFrame with features.
        feature_columns: Feature columns to analyze.
        compute_umap: Whether to compute UMAP (requires umap-learn).
        generate_plots: Whether to generate visualizations.

    Returns:
        Dictionary with all analysis results.
    """
    ensure_directories()

    all_results: dict[str, Any] = {}
    total_start = time.time()

    # 1. CORRELATION ANALYSIS
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS 1/5: CORRELATION ANALYSIS")
    logger.info("=" * 70)

    start = time.time()
    corr_results = run_correlation_analysis(df, feature_columns)
    all_results["correlation"] = corr_results
    logger.info("Correlation analysis completed in %.1f seconds", time.time() - start)

    if generate_plots and "spearman" in corr_results:
        plot_correlation_heatmap(
            corr_results["spearman"],
            title="Spearman Correlation Matrix",
            filename="spearman_correlation",
        )

    # 2. MULTICOLLINEARITY ANALYSIS
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS 2/5: MULTICOLLINEARITY ANALYSIS")
    logger.info("=" * 70)

    start = time.time()
    multicol_results = run_multicollinearity_analysis(df, feature_columns)
    all_results["multicollinearity"] = multicol_results
    logger.info("Multicollinearity analysis completed in %.1f seconds", time.time() - start)

    if generate_plots and "vif" in multicol_results:
        plot_vif_scores(multicol_results["vif"])

    # 3. TARGET RELATIONSHIP ANALYSIS
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS 3/5: TARGET RELATIONSHIP ANALYSIS")
    logger.info("=" * 70)

    start = time.time()
    if TARGET_COLUMN in df.columns:
        target_results = run_target_analysis(df, TARGET_COLUMN, feature_columns)
        all_results["target"] = target_results

        if generate_plots and "combined_metrics" in target_results:
            plot_target_correlations(target_results["combined_metrics"])
    else:
        logger.warning("Target column '%s' not found, skipping target analysis", TARGET_COLUMN)
    logger.info("Target analysis completed in %.1f seconds", time.time() - start)

    # 4. CLUSTERING ANALYSIS
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS 4/5: CLUSTERING ANALYSIS")
    logger.info("=" * 70)

    start = time.time()
    cluster_results = run_clustering_analysis(
        df,
        feature_columns,
        n_clusters=None,  # Auto-select optimal k via silhouette analysis
        min_clusters=2,
        max_clusters=10,
        compute_tsne=True,
        compute_umap=compute_umap,
        detect_outliers=True,
    )
    all_results["clustering"] = cluster_results
    logger.info("Clustering analysis completed in %.1f seconds", time.time() - start)

    if generate_plots:
        _generate_clustering_plots(cluster_results, feature_columns)

    # 5. TEMPORAL ANALYSIS
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS 5/5: TEMPORAL ANALYSIS")
    logger.info("=" * 70)

    start = time.time()
    temporal_results = run_temporal_analysis(df, feature_columns)
    all_results["temporal"] = temporal_results
    logger.info("Temporal analysis completed in %.1f seconds", time.time() - start)

    # SUMMARY
    total_time = time.time() - total_start
    _save_full_summary(all_results, total_time)

    logger.info("\n" + "=" * 70)
    logger.info("FEATURE ANALYSIS COMPLETE")
    logger.info("=" * 70)
    logger.info("Total analysis time: %.1f seconds (%.1f minutes)", total_time, total_time / 60)
    logger.info("Results saved to: %s", ANALYSE_FEATURES_DIR)

    return all_results


def print_summary(results: dict[str, Any]) -> None:
    """Print analysis summary to console."""
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS SUMMARY")
    logger.info("=" * 70)

    if "correlation" in results and "spearman" in results["correlation"]:
        spearman = results["correlation"]["spearman"]
        n_features = len(spearman)
        high_corr = ((spearman.abs() > 0.7) & (spearman.abs() < 1.0)).sum().sum() // 2
        logger.info("Correlation: %d features, %d high correlation pairs (>0.7)", n_features, high_corr)


    if "multicollinearity" in results and "vif" in results["multicollinearity"]:
        vif_df = results["multicollinearity"]["vif"]
        high_vif = (vif_df["vif"] > 10).sum()
        logger.info("Multicollinearity: %d features with VIF > 10", high_vif)

    if "target" in results and "summary" in results["target"]:
        summary = results["target"]["summary"]
        logger.info(
            "Target: avg |Spearman| = %.4f, avg MI = %.4f",
            summary.get("avg_abs_spearman", 0),
            summary.get("avg_mi", 0),
        )

    if "clustering" in results and "clusters" in results["clustering"]:
        clusters_df = results["clustering"]["clusters"]
        n_clusters = clusters_df["cluster"].nunique()
        cluster_results = results["clustering"]

        # Get quality metrics
        if "final_metrics" in cluster_results:
            metrics = cluster_results["final_metrics"]
            sil = metrics.get("silhouette_score", float("nan"))
            db = metrics.get("davies_bouldin_score", float("nan"))
            logger.info(
                "Clustering: %d clusters (silhouette=%.3f, davies_bouldin=%.3f)",
                n_clusters,
                sil,
                db,
            )
        else:
            logger.info("Clustering: %d clusters identified", n_clusters)

        # Report interpretation
        if "cluster_interpretation" in cluster_results:
            logger.info("  Interpretation: %s", cluster_results["cluster_interpretation"])

        # Report outliers
        if "tsne_outliers" in cluster_results:
            logger.warning("  t-SNE outliers: %s", cluster_results["tsne_outliers"])
        if "umap_outliers" in cluster_results:
            logger.warning("  UMAP outliers: %s", cluster_results["umap_outliers"])


def main() -> None:
    """Main entry point for feature analysis."""
    parser = argparse.ArgumentParser(
        description="Feature Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--input", type=str, default=None, help="Path to features parquet file")
    parser.add_argument(
        "--analysis",
        type=str,
        choices=["all", "correlation", "multicollinearity", "target", "clustering", "temporal"],
        default="all",
        help="Which analysis to run (default: all)",
    )
    parser.add_argument("--skip-umap", action="store_true", help="Skip UMAP embedding")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--use-all-data", action="store_true", help="Use both train and test data")

    args = parser.parse_args()

    setup_logging()

    logger.info("=" * 70)
    logger.info("FEATURE ANALYSIS PIPELINE")
    logger.info("=" * 70)

    input_path = Path(args.input) if args.input else None
    df, feature_columns = load_features(file_path=input_path, use_train_only=not args.use_all_data)

    logger.info("Loaded %d rows, %d features", len(df), len(feature_columns))

    if args.analysis == "all":
        results = run_all_analyses(
            df,
            feature_columns,
            compute_umap=not args.skip_umap,
            generate_plots=not args.no_plots,
        )
        print_summary(results)

    elif args.analysis == "correlation":
        ensure_directories()
        results = run_correlation_analysis(df, feature_columns)
        if not args.no_plots and "spearman" in results:
            plot_correlation_heatmap(results["spearman"], title="Spearman Correlation", filename="spearman_correlation")

    elif args.analysis == "multicollinearity":
        ensure_directories()
        results = run_multicollinearity_analysis(df, feature_columns)
        if not args.no_plots and "vif" in results:
            plot_vif_scores(results["vif"])

    elif args.analysis == "target":
        ensure_directories()
        results = run_target_analysis(df, TARGET_COLUMN, feature_columns)
        if not args.no_plots and "combined_metrics" in results:
            plot_target_correlations(results["combined_metrics"])

    elif args.analysis == "clustering":
        ensure_directories()
        run_clustering_analysis(
            df,
            feature_columns,
            n_clusters=None,  # Auto-select optimal k
            min_clusters=2,
            max_clusters=10,
            compute_umap=not args.skip_umap,
            detect_outliers=True,
        )

    elif args.analysis == "temporal":   
        ensure_directories()
        run_temporal_analysis(df, feature_columns)

    logger.info("\nAnalysis complete!")


if __name__ == "__main__":
    main()
