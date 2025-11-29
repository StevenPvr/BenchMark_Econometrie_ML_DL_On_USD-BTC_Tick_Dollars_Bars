import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.analyse_features.clustering import (
    compute_correlation_distance_matrix,
    hierarchical_clustering,
    cut_dendrogram,
    compute_tsne_embedding,
    compute_umap_embedding,
    run_clustering_analysis
)

class TestClustering:

    def test_compute_correlation_distance_matrix(self, correlated_df):
        cols = ["feat_a", "feat_b", "feat_c"]
        dist_matrix, corr_matrix = compute_correlation_distance_matrix(correlated_df, cols)

        assert dist_matrix.shape == (3, 3)
        assert corr_matrix.shape == (3, 3)

        # Diagonal distance should be 0
        np.testing.assert_allclose(np.diag(dist_matrix), 0.0)

        # Feat A and B highly correlated -> low distance
        idx_a, idx_b = 0, 1
        assert dist_matrix[idx_a, idx_b] < 0.2

    def test_hierarchical_clustering(self, correlated_df):
        cols = ["feat_a", "feat_b", "feat_c"]
        dist_matrix, _ = compute_correlation_distance_matrix(correlated_df, cols)

        result = hierarchical_clustering(dist_matrix, cols)

        assert "linkage_matrix" in result
        assert "dendrogram" in result

        # Linkage matrix should be (n-1) x 4
        assert result["linkage_matrix"].shape == (2, 4)

    def test_cut_dendrogram(self, correlated_df):
        cols = ["feat_a", "feat_b", "feat_c"]
        dist_matrix, _ = compute_correlation_distance_matrix(correlated_df, cols)
        hier_res = hierarchical_clustering(dist_matrix, cols)
        linkage = hier_res["linkage_matrix"]

        # Cut into 2 clusters (correlated pair + uncorrelated one)
        clusters = cut_dendrogram(linkage, cols, n_clusters=2)

        assert "cluster" in clusters.columns
        assert clusters["cluster"].nunique() == 2

        # feat_a and feat_b should likely be in same cluster
        cluster_a = clusters[clusters["feature"] == "feat_a"]["cluster"].values[0]
        cluster_b = clusters[clusters["feature"] == "feat_b"]["cluster"].values[0]
        assert cluster_a == cluster_b

    def test_compute_tsne_embedding(self, correlated_df):
        # Small perplexity for small dataset
        cols = ["feat_a", "feat_b", "feat_c"]
        tsne_df = compute_tsne_embedding(correlated_df, cols, perplexity=2, n_iter=250)

        assert "tsne_1" in tsne_df.columns
        assert "tsne_2" in tsne_df.columns
        assert len(tsne_df) == 3

    def test_compute_umap_embedding_no_umap(self, correlated_df):
        # Assume UMAP not installed or check availability
        from src.analyse_features.clustering import UMAP_AVAILABLE

        cols = ["feat_a", "feat_b", "feat_c"]
        umap_df = compute_umap_embedding(correlated_df, cols)

        if not UMAP_AVAILABLE:
            assert umap_df["umap_1"].isna().all()
        else:
            assert not umap_df["umap_1"].isna().all()

    @patch("src.analyse_features.clustering.save_json")
    @patch("src.analyse_features.clustering.plot_dendrogram")
    @patch("src.analyse_features.clustering.plot_embedding")
    @patch("src.analyse_features.clustering.ensure_directories")
    def test_run_clustering_analysis(self, mock_ensure, mock_plot_emb, mock_plot_dendro, mock_save, correlated_df):
        results = run_clustering_analysis(correlated_df, n_clusters=2, save_results=True)

        assert "clusters" in results
        assert "families" in results
        assert "distance_matrix" in results

        mock_ensure.assert_called()
        mock_save.assert_called()
