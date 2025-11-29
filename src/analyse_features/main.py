"""Main orchestrator for feature analysis pipeline.

This module coordinates all feature analysis components:
1. Correlation Analysis (Spearman, dCor, MI)
2. Stationarity Analysis (ADF, KPSS)
3. Multicollinearity Analysis (VIF, Condition Number)
4. Target Relationship Analysis
5. Feature Clustering (Hierarchical, t-SNE, UMAP)
6. Temporal Analysis (ACF/PACF, Rolling Correlations)

Usage:
    # Run all analyses
    python -m src.analyse_features.main

    # Run specific analysis
    python -m src.analyse_features.main --analysis correlation

    # Skip slow analyses
    python -m src.analyse_features.main --skip-dcor --skip-umap
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

# Add project root to path
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config_logging import get_logger, setup_logging  # noqa: E402
from src.constants import DEFAULT_RANDOM_STATE  # noqa: E402
from src.path import DATASET_FEATURES_PARQUET  # noqa: E402

from src.analyse_features.config import (  # noqa: E402
    ANALYSE_FEATURES_DIR,
    TARGET_COLUMN,
    CORRELATION_RESULTS_JSON,
    STATIONARITY_RESULTS_JSON,
    MULTICOLLINEARITY_RESULTS_JSON,
    TARGET_RESULTS_JSON,
    CLUSTERING_RESULTS_JSON,
    TEMPORAL_RESULTS_JSON,
    SUMMARY_JSON,
    DATASET_SAMPLE_FRACTION,
    ensure_directories,
)
from src.analyse_features.correlation import run_correlation_analysis  # noqa: E402
from src.analyse_features.stationarity import run_stationarity_analysis  # noqa: E402
from src.analyse_features.multicollinearity import run_multicollinearity_analysis  # noqa: E402
from src.analyse_features.target_analysis import run_target_analysis  # noqa: E402
from src.analyse_features.clustering import run_clustering_analysis  # noqa: E402
from src.analyse_features.temporal import run_temporal_analysis  # noqa: E402
from src.analyse_features.utils.plotting import (  # noqa: E402
    plot_correlation_heatmap,
    plot_vif_scores,
    plot_dendrogram,
    plot_embedding,
    plot_stationarity_summary,
    plot_target_correlations,
)

logger = get_logger(__name__)


def _convert_to_serializable(obj: Any) -> Any:
    """Convert numpy/pandas types to JSON-serializable Python types."""
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    elif isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {
            _convert_to_serializable(k): _convert_to_serializable(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [_convert_to_serializable(v) for v in obj]
    elif isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(data: dict[str, Any], filepath: Path) -> None:
    """Save dictionary to JSON file."""
    serializable_data = _convert_to_serializable(data)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)
    logger.info("Saved JSON: %s", filepath)


def save_correlation_results(results: dict[str, Any]) -> None:
    """Save correlation analysis results to JSON."""
    json_data: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "analysis": "correlation",
    }

    # Spearman summary (top correlations, not full matrix)
    if "spearman" in results:
        spearman = results["spearman"]
        n_features = len(spearman)

        # Get top correlated pairs
        corr_pairs = []
        for i in range(len(spearman)):
            for j in range(i + 1, len(spearman)):
                val = spearman.iloc[i, j]
                if abs(val) > 0.5:  # Only significant correlations
                    corr_pairs.append({
                        "feature_1": spearman.index[i],
                        "feature_2": spearman.columns[j],
                        "correlation": float(val),
                    })

        corr_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        json_data["spearman"] = {
            "n_features": n_features,
            "high_correlation_pairs": corr_pairs[:100],  # Top 100
            "n_high_corr_pairs": len(corr_pairs),
        }

    # Mutual information
    if "mutual_information" in results:
        mi_df = results["mutual_information"]
        json_data["mutual_information"] = {
            "top_features": mi_df.head(50).to_dict(orient="records"),
        }

    save_json(json_data, CORRELATION_RESULTS_JSON)


def save_stationarity_results(results: pd.DataFrame) -> None:
    """Save stationarity analysis results to JSON."""
    summary = results["stationarity_conclusion"].value_counts().to_dict()

    json_data: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "analysis": "stationarity",
        "summary": summary,
        "stationary_features": results[
            results["stationarity_conclusion"] == "stationary"
        ]["feature"].tolist(),
        "non_stationary_features": results[
            results["stationarity_conclusion"] == "non_stationary"
        ]["feature"].tolist(),
        "all_results": results.to_dict(orient="records"),
    }

    save_json(json_data, STATIONARITY_RESULTS_JSON)


def save_multicollinearity_results(results: dict[str, Any]) -> None:
    """Save multicollinearity analysis results to JSON."""
    json_data: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "analysis": "multicollinearity",
    }

    if "vif" in results:
        vif_df = results["vif"]
        json_data["vif"] = {
            "high_vif_features": vif_df[vif_df["vif"] > 10]["feature"].tolist(),
            "critical_vif_features": vif_df[vif_df["vif"] > 100]["feature"].tolist(),
            "all_scores": vif_df.to_dict(orient="records"),
        }

    if "condition_number" in results:
        json_data["condition_number"] = results["condition_number"]

    if "collinear_pairs" in results:
        pairs_df = results["collinear_pairs"]
        if not pairs_df.empty:
            json_data["collinear_pairs"] = pairs_df.head(50).to_dict(orient="records")

    if "drop_suggestions" in results:
        json_data["drop_suggestions"] = results["drop_suggestions"]

    save_json(json_data, MULTICOLLINEARITY_RESULTS_JSON)


def save_target_results(results: dict[str, Any]) -> None:
    """Save target analysis results to JSON."""
    json_data: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "analysis": "target_relationship",
    }

    if "combined_metrics" in results:
        metrics_df = results["combined_metrics"]
        json_data["top_features"] = metrics_df.head(50).to_dict(orient="records")
        json_data["feature_rankings"] = {
            "by_pearson": metrics_df.nsmallest(30, "rank_pearson")["feature"].tolist(),
            "by_spearman": metrics_df.nsmallest(30, "rank_spearman")["feature"].tolist(),
            "by_mi": metrics_df.nsmallest(30, "rank_mi")["feature"].tolist(),
        }

    if "summary" in results:
        json_data["summary"] = results["summary"]

    save_json(json_data, TARGET_RESULTS_JSON)


def save_clustering_results(results: dict[str, Any]) -> None:
    """Save clustering analysis results to JSON."""
    json_data: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "analysis": "clustering",
    }

    if "clusters" in results:
        clusters_df = results["clusters"]
        n_clusters: Any = int(clusters_df["cluster"].nunique())
        json_data["n_clusters"] = n_clusters
        cluster_assignments: Any = clusters_df.to_dict(orient="records")
        json_data["cluster_assignments"] = cluster_assignments

    if "families" in results:
        json_data["families"] = results["families"]

    if "tsne" in results:
        tsne_computed: Any = True
        json_data["tsne_computed"] = tsne_computed

    if "umap" in results:
        umap_computed: Any = not results["umap"]["umap_1"].isna().all()
        json_data["umap_computed"] = umap_computed

    save_json(json_data, CLUSTERING_RESULTS_JSON)


def save_temporal_results(results: dict[str, Any]) -> None:
    """Save temporal analysis results to JSON."""
    json_data: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "analysis": "temporal",
    }

    if "acf_summary" in results:
        acf_df = results["acf_summary"]
        json_data["most_persistent"] = acf_df.head(30).to_dict(orient="records")

    if "stability" in results:
        stab_df = results["stability"]
        json_data["most_stable"] = stab_df.head(30).to_dict(orient="records")

    if "correlation_over_time" in results:
        corr_df = results["correlation_over_time"]
        json_data["most_consistent"] = corr_df.head(30).to_dict(orient="records")

    save_json(json_data, TEMPORAL_RESULTS_JSON)


def save_full_summary(all_results: dict[str, Any], total_time: float) -> None:
    """Save complete analysis summary."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_time_seconds": total_time,
        "analyses_completed": list(all_results.keys()),
    }

    # Correlation summary
    if "correlation" in all_results and "spearman" in all_results["correlation"]:
        spearman = all_results["correlation"]["spearman"]
        high_corr = ((spearman.abs() > 0.7) & (spearman.abs() < 1.0)).sum().sum() // 2
        summary["correlation"] = {
            "n_features": len(spearman),
            "high_correlation_pairs": int(high_corr),
        }

    # Stationarity summary
    if "stationarity" in all_results:
        station_df = all_results["stationarity"]
        summary["stationarity"] = station_df["stationarity_conclusion"].value_counts().to_dict()

    # Multicollinearity summary
    if "multicollinearity" in all_results and "vif" in all_results["multicollinearity"]:
        vif_df = all_results["multicollinearity"]["vif"]
        summary["multicollinearity"] = {
            "features_vif_over_10": int((vif_df["vif"] > 10).sum()),
            "features_vif_over_100": int((vif_df["vif"] > 100).sum()),
        }

    # Target summary
    if "target" in all_results and "summary" in all_results["target"]:
        summary["target"] = all_results["target"]["summary"]

    # Clustering summary
    if "clustering" in all_results and "clusters" in all_results["clustering"]:
        clusters_df = all_results["clustering"]["clusters"]
        summary["clustering"] = {
            "n_clusters": int(clusters_df["cluster"].nunique()),
        }

    save_json(summary, SUMMARY_JSON)


def load_features(
    file_path: Path | None = None,
    use_train_only: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Load feature dataset and extract feature columns.

    Args:
        file_path: Path to features parquet (default: DATASET_FEATURES_PARQUET).
        use_train_only: Whether to use only training data.

    Returns:
        Tuple of (DataFrame, list of feature column names).
    """
    file_path = file_path or DATASET_FEATURES_PARQUET

    logger.info("Loading features from %s", file_path)
    df = pd.read_parquet(file_path)

    initial_rows = len(df)

    # Filter to train split if requested
    if use_train_only and "split" in df.columns:
        df = df[df["split"] == "train"].copy()
        df = df.drop(columns=["split"])
        logger.info("Filtered to train split: %d -> %d rows", initial_rows, len(df))

    # Sample fraction of dataset for analysis
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

    # Get feature columns (numeric, excluding target and metadata)
    exclude_cols = {TARGET_COLUMN, "split", "index", "timestamp", "datetime", "date"}
    numeric_cols = df.select_dtypes(include=["number"]).columns
    feature_columns: list[str] = [
        str(col) for col in numeric_cols
        if col not in exclude_cols
    ]

    logger.info("Found %d feature columns", len(feature_columns))

    return cast(pd.DataFrame, df), feature_columns


def run_all_analyses(
    df: pd.DataFrame,
    feature_columns: list[str],
    compute_dcor: bool = True,
    compute_umap: bool = True,
    generate_plots: bool = True,
) -> dict[str, Any]:
    """Run complete feature analysis pipeline.

    Args:
        df: DataFrame with features.
        feature_columns: Feature columns to analyze.
        compute_dcor: Whether to compute distance correlation (deprecated, ignored).
        compute_umap: Whether to compute UMAP (requires umap-learn).
        generate_plots: Whether to generate visualizations.

    Returns:
        Dictionary with all analysis results.
    """
    ensure_directories()

    all_results = {}
    total_start = time.time()

    # =========================================================================
    # 1. CORRELATION ANALYSIS
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS 1/6: CORRELATION ANALYSIS")
    logger.info("=" * 70)

    start = time.time()
    corr_results = run_correlation_analysis(
        df,
        feature_columns,
        compute_dcor=compute_dcor,
    )
    all_results["correlation"] = corr_results
    save_correlation_results(corr_results)
    logger.info("Correlation analysis completed in %.1f seconds", time.time() - start)

    if generate_plots and "spearman" in corr_results:
        plot_correlation_heatmap(
            corr_results["spearman"],
            title="Spearman Correlation Matrix",
            filename="spearman_correlation",
        )

    # =========================================================================
    # 2. STATIONARITY ANALYSIS
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS 2/6: STATIONARITY ANALYSIS")
    logger.info("=" * 70)

    start = time.time()
    station_results = run_stationarity_analysis(df, feature_columns)
    all_results["stationarity"] = station_results
    save_stationarity_results(station_results)
    logger.info("Stationarity analysis completed in %.1f seconds", time.time() - start)

    if generate_plots:
        plot_stationarity_summary(station_results)

    # =========================================================================
    # 3. MULTICOLLINEARITY ANALYSIS
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS 3/6: MULTICOLLINEARITY ANALYSIS")
    logger.info("=" * 70)

    start = time.time()
    multicol_results = run_multicollinearity_analysis(df, feature_columns)
    all_results["multicollinearity"] = multicol_results
    save_multicollinearity_results(multicol_results)
    logger.info("Multicollinearity analysis completed in %.1f seconds", time.time() - start)

    if generate_plots and "vif" in multicol_results:
        plot_vif_scores(multicol_results["vif"])

    # =========================================================================
    # 4. TARGET RELATIONSHIP ANALYSIS
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS 4/6: TARGET RELATIONSHIP ANALYSIS")
    logger.info("=" * 70)

    start = time.time()
    if TARGET_COLUMN in df.columns:
        target_results = run_target_analysis(df, TARGET_COLUMN, feature_columns)
        all_results["target"] = target_results
        save_target_results(target_results)

        if generate_plots and "combined_metrics" in target_results:
            plot_target_correlations(target_results["combined_metrics"])
    else:
        logger.warning("Target column '%s' not found, skipping target analysis", TARGET_COLUMN)
    logger.info("Target analysis completed in %.1f seconds", time.time() - start)

    # =========================================================================
    # 5. CLUSTERING ANALYSIS
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS 5/6: CLUSTERING ANALYSIS")
    logger.info("=" * 70)

    start = time.time()
    cluster_results = run_clustering_analysis(
        df,
        feature_columns,
        n_clusters=15,
        compute_tsne=True,
        compute_umap=compute_umap,
    )
    all_results["clustering"] = cluster_results
    save_clustering_results(cluster_results)
    logger.info("Clustering analysis completed in %.1f seconds", time.time() - start)

    if generate_plots:
        # Dendrogram
        if "hierarchical" in cluster_results:
            plot_dendrogram(
                cluster_results["hierarchical"]["linkage_matrix"],
                feature_columns,
                title="Feature Hierarchical Clustering",
                filename="feature_dendrogram",
            )

        # t-SNE embedding
        if "tsne" in cluster_results and "clusters" in cluster_results:
            tsne_df = cluster_results["tsne"].merge(
                cluster_results["clusters"],
                on="feature",
            )
            plot_embedding(
                tsne_df,
                x_col="tsne_1",
                y_col="tsne_2",
                label_col="feature",
                color_col="cluster",
                title="t-SNE Feature Embedding",
                filename="tsne_embedding",
            )

        # UMAP embedding
        if "umap" in cluster_results and "clusters" in cluster_results:
            umap_df = cluster_results["umap"].merge(
                cluster_results["clusters"],
                on="feature",
            )
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

    # =========================================================================
    # 6. TEMPORAL ANALYSIS
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS 6/6: TEMPORAL ANALYSIS")
    logger.info("=" * 70)

    start = time.time()
    temporal_results = run_temporal_analysis(df, feature_columns)
    all_results["temporal"] = temporal_results
    save_temporal_results(temporal_results)
    logger.info("Temporal analysis completed in %.1f seconds", time.time() - start)

    # =========================================================================
    # SUMMARY & SAVE ALL
    # =========================================================================
    total_time = time.time() - total_start

    # Save complete summary JSON
    save_full_summary(all_results, total_time)

    logger.info("\n" + "=" * 70)
    logger.info("FEATURE ANALYSIS COMPLETE")
    logger.info("=" * 70)
    logger.info("Total analysis time: %.1f seconds (%.1f minutes)", total_time, total_time / 60)
    logger.info("Results saved to: %s", ANALYSE_FEATURES_DIR)
    logger.info("JSON files:")
    logger.info("  - %s", SUMMARY_JSON)
    logger.info("  - %s", CORRELATION_RESULTS_JSON)
    logger.info("  - %s", STATIONARITY_RESULTS_JSON)
    logger.info("  - %s", MULTICOLLINEARITY_RESULTS_JSON)
    logger.info("  - %s", TARGET_RESULTS_JSON)
    logger.info("  - %s", CLUSTERING_RESULTS_JSON)
    logger.info("  - %s", TEMPORAL_RESULTS_JSON)

    return all_results


def print_summary(results: dict[str, Any]) -> None:
    """Print analysis summary to console.

    Args:
        results: Dictionary with all analysis results.
    """
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS SUMMARY")
    logger.info("=" * 70)

    # Correlation summary
    if "correlation" in results and "spearman" in results["correlation"]:
        spearman = results["correlation"]["spearman"]
        n_features = len(spearman)
        high_corr = ((spearman.abs() > 0.7) & (spearman.abs() < 1.0)).sum().sum() // 2
        logger.info("Correlation: %d features, %d high correlation pairs (>0.7)", n_features, high_corr)

    # Stationarity summary
    if "stationarity" in results:
        station_df = results["stationarity"]
        stationary_count = (station_df["stationarity_conclusion"] == "stationary").sum()
        non_stationary_count = (station_df["stationarity_conclusion"] == "non_stationary").sum()
        logger.info("Stationarity: %d stationary, %d non-stationary", stationary_count, non_stationary_count)

    # VIF summary
    if "multicollinearity" in results and "vif" in results["multicollinearity"]:
        vif_df = results["multicollinearity"]["vif"]
        high_vif = (vif_df["vif"] > 10).sum()
        logger.info("Multicollinearity: %d features with VIF > 10", high_vif)

    # Target summary
    if "target" in results and "summary" in results["target"]:
        summary = results["target"]["summary"]
        logger.info("Target: avg |Spearman| = %.4f, avg MI = %.4f",
                   summary.get("avg_abs_spearman", 0),
                   summary.get("avg_mi", 0))

    # Clustering summary
    if "clustering" in results and "clusters" in results["clustering"]:
        clusters_df = results["clustering"]["clusters"]
        n_clusters = clusters_df["cluster"].nunique()
        logger.info("Clustering: %d clusters identified", n_clusters)


def main() -> None:
    """Main entry point for feature analysis."""
    parser = argparse.ArgumentParser(
        description="Feature Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to features parquet file",
    )

    parser.add_argument(
        "--analysis",
        type=str,
        choices=["all", "correlation", "stationarity", "multicollinearity",
                 "target", "clustering", "temporal"],
        default="all",
        help="Which analysis to run (default: all)",
    )

    parser.add_argument(
        "--skip-dcor",
        action="store_true",
        help="Skip distance correlation computation (slow)",
    )

    parser.add_argument(
        "--skip-umap",
        action="store_true",
        help="Skip UMAP embedding (requires umap-learn)",
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation",
    )

    parser.add_argument(
        "--use-all-data",
        action="store_true",
        help="Use both train and test data (default: train only)",
    )

    args = parser.parse_args()

    setup_logging()

    logger.info("=" * 70)
    logger.info("FEATURE ANALYSIS PIPELINE")
    logger.info("=" * 70)

    # Load data
    input_path = Path(args.input) if args.input else None
    df, feature_columns = load_features(
        file_path=input_path,
        use_train_only=not args.use_all_data,
    )

    logger.info("Loaded %d rows, %d features", len(df), len(feature_columns))

    # Run analyses
    if args.analysis == "all":
        results = run_all_analyses(
            df,
            feature_columns,
            compute_dcor=not args.skip_dcor,
            compute_umap=not args.skip_umap,
            generate_plots=not args.no_plots,
        )
        print_summary(results)

    elif args.analysis == "correlation":
        ensure_directories()
        results = run_correlation_analysis(df, feature_columns, compute_dcor=not args.skip_dcor)
        save_correlation_results(results)
        if not args.no_plots and "spearman" in results:
            plot_correlation_heatmap(results["spearman"], title="Spearman Correlation", filename="spearman_correlation")

    elif args.analysis == "stationarity":
        ensure_directories()
        results = run_stationarity_analysis(df, feature_columns)
        save_stationarity_results(results)
        if not args.no_plots:
            plot_stationarity_summary(results)

    elif args.analysis == "multicollinearity":
        ensure_directories()
        results = run_multicollinearity_analysis(df, feature_columns)
        save_multicollinearity_results(results)
        if not args.no_plots and "vif" in results:
            plot_vif_scores(results["vif"])

    elif args.analysis == "target":
        ensure_directories()
        results = run_target_analysis(df, TARGET_COLUMN, feature_columns)
        save_target_results(results)
        if not args.no_plots and "combined_metrics" in results:
            plot_target_correlations(results["combined_metrics"])

    elif args.analysis == "clustering":
        ensure_directories()
        results = run_clustering_analysis(df, feature_columns, compute_umap=not args.skip_umap)
        save_clustering_results(results)

    elif args.analysis == "temporal":
        ensure_directories()
        results = run_temporal_analysis(df, feature_columns)
        save_temporal_results(results)

    logger.info("\nAnalysis complete!")


if __name__ == "__main__":
    main()
