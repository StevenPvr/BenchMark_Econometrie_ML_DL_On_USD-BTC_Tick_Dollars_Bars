import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.analyse_features.utils.plotting import (
    plot_correlation_heatmap,
    plot_vif_scores,
    plot_dendrogram,
    plot_embedding,
    plot_stationarity_summary,
    plot_target_correlations
)

class TestPlotting:

    @patch("src.analyse_features.utils.plotting.plt")
    def test_plot_correlation_heatmap(self, mock_plt):
        # Configure subplots to return fig, ax
        mock_plt.subplots.return_value = (MagicMock(), MagicMock())

        df = pd.DataFrame(np.eye(3), columns=["a", "b", "c"], index=["a", "b", "c"])
        plot_correlation_heatmap(df, title="Test", filename="test")

        assert mock_plt.subplots.called or mock_plt.figure.called

    @patch("src.analyse_features.utils.plotting.plt")
    def test_plot_vif_scores(self, mock_plt):
        mock_plt.subplots.return_value = (MagicMock(), MagicMock())
        df = pd.DataFrame({"feature": ["a", "b"], "vif": [1.5, 2.5]})
        plot_vif_scores(df)
        assert mock_plt.subplots.called or mock_plt.figure.called

    @patch("src.analyse_features.utils.plotting.plt")
    @patch("src.analyse_features.utils.plotting.hierarchy.dendrogram")
    def test_plot_dendrogram(self, mock_dendro, mock_plt):
        mock_plt.subplots.return_value = (MagicMock(), MagicMock())
        linkage = np.zeros((2, 4))
        features = ["a", "b", "c"]
        plot_dendrogram(linkage, features, title="Test", filename="test")
        mock_plt.subplots.called

    @patch("src.analyse_features.utils.plotting.plt")
    def test_plot_embedding(self, mock_plt):
        mock_plt.subplots.return_value = (MagicMock(), MagicMock())
        df = pd.DataFrame({
            "x": [1, 2], "y": [1, 2],
            "label": ["a", "b"], "color": [0, 1]
        })
        plot_embedding(df, "x", "y", "label", "color", title="Test", filename="test")
        mock_plt.subplots.called

    @patch("src.analyse_features.utils.plotting.plt")
    def test_plot_stationarity_summary(self, mock_plt):
        mock_plt.subplots.return_value = (MagicMock(), MagicMock())
        df = pd.DataFrame({
            "feature": ["a", "b"],
            "stationarity_conclusion": ["stationary", "non_stationary"]
        })
        plot_stationarity_summary(df)
        mock_plt.subplots.called

    @patch("src.analyse_features.utils.plotting.plt")
    def test_plot_target_correlations(self, mock_plt):
        mock_plt.subplots.return_value = (MagicMock(), MagicMock())
        df = pd.DataFrame({
            "feature": ["a", "b"],
            "abs_spearman": [0.8, 0.5],
            "abs_pearson": [0.7, 0.4], # Added missing column
            "mutual_information": [0.5, 0.2]
        })
        plot_target_correlations(df)
        mock_plt.subplots.called
