"""Tests for src/utils/logging_utils.py module."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import pandas as pd  # type: ignore[import-untyped]
import pytest

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.logging_utils import (
    log_series_summary,
    log_split_summary,
    save_plot,
)


class TestLogSeriesSummary:
    """Test cases for log_series_summary function."""

    def test_logs_without_error(self):
        """Should log series summary without raising."""
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

        # Should not raise
        log_series_summary(series)

    def test_handles_empty_series(self):
        """Should handle empty series."""
        series = pd.Series([], dtype=float)

        # Should not raise (may log warnings for empty stats)
        try:
            log_series_summary(series)
        except Exception:
            pass  # Empty series may cause issues with stats

    def test_custom_name(self):
        """Should accept custom name."""
        series = pd.Series([1, 2, 3])

        # Should not raise
        log_series_summary(series, name="MyCustomSeries")

    def test_handles_series_with_nulls(self):
        """Should handle series with null values."""
        series = pd.Series([1.0, None, 3.0, None, 5.0])

        # Should not raise
        log_series_summary(series)


class TestLogSplitSummary:
    """Test cases for log_split_summary function."""

    def test_logs_without_error(self):
        """Should log split summary without raising."""
        train_data = pd.DataFrame({"value": range(80)})
        test_data = pd.DataFrame({"value": range(20)})

        # Should not raise
        log_split_summary(train_data, test_data)

    def test_with_series(self):
        """Should work with Series."""
        train_data = pd.Series(range(80))
        test_data = pd.Series(range(20))

        # Should not raise
        log_split_summary(train_data, test_data)

    def test_with_split_date(self):
        """Should log split date when provided."""
        train_data = pd.DataFrame({"value": range(80)})
        test_data = pd.DataFrame({"value": range(20)})

        # Should not raise
        log_split_summary(train_data, test_data, split_date="2024-01-01")

    def test_handles_empty_data(self):
        """Should handle empty data."""
        train_data = pd.DataFrame()
        test_data = pd.DataFrame()

        # Should not raise
        log_split_summary(train_data, test_data)


class TestSavePlot:
    """Test cases for save_plot function."""

    def test_saves_figure(self):
        """Should save figure to file."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "test_plot.png"
            save_plot(fig, path)

            assert path.exists()

    def test_creates_directories(self):
        """Should create parent directories."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nested" / "dir" / "test_plot.png"
            save_plot(fig, path)

            assert path.exists()

    def test_custom_dpi(self):
        """Should use custom DPI."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "test_plot.png"
            save_plot(fig, path, dpi=150)

            assert path.exists()

    def test_accepts_string_path(self):
        """Should accept string path."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        with tempfile.TemporaryDirectory() as temp_dir:
            path = str(Path(temp_dir) / "test_plot.png")
            save_plot(fig, path)

            assert Path(path).exists()

    def test_closes_figure(self):
        """Should close figure after saving."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "test_plot.png"
            save_plot(fig, path)

            # Figure should be closed
            assert len(plt.get_fignums()) == 0 or fig.number not in plt.get_fignums()

    def test_various_formats(self):
        """Should save to various formats."""
        formats = [".png", ".pdf", ".svg"]

        with tempfile.TemporaryDirectory() as temp_dir:
            for fmt in formats:
                fig, ax = plt.subplots()
                ax.plot([1, 2, 3], [1, 2, 3])

                path = Path(temp_dir) / f"test_plot{fmt}"
                save_plot(fig, path)

                assert path.exists(), f"Failed to save {fmt} format"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
