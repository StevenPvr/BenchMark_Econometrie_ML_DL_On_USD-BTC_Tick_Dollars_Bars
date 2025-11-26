"""Tests for src/garch/garch_eval/helpers.py module."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.garch.garch_eval.helpers import (
    _validate_and_filter_arrays,
    _clip_probability,
    parse_alphas,
    to_numpy,
)


class TestValidateAndFilterArrays:
    """Test cases for _validate_and_filter_arrays function."""

    def test_filters_non_finite(self):
        """Should filter out non-finite values."""
        e = np.array([1.0, np.nan, 3.0, np.inf])
        sigma2 = np.array([0.1, 0.2, 0.3, 0.4])

        e_filt, s2_filt, mask = _validate_and_filter_arrays(e, sigma2)

        assert len(e_filt) == 2  # Only indices 0 and 2 are valid
        assert len(s2_filt) == 2

    def test_filters_non_positive_variance(self):
        """Should filter out non-positive variance."""
        e = np.array([1.0, 2.0, 3.0, 4.0])
        sigma2 = np.array([0.1, -0.1, 0.3, 0.0])

        e_filt, s2_filt, mask = _validate_and_filter_arrays(e, sigma2)

        assert len(e_filt) == 2  # Only indices 0 and 2 have positive variance

    def test_returns_mask(self):
        """Should return boolean mask."""
        e = np.array([1.0, 2.0])
        sigma2 = np.array([0.1, 0.2])

        e_filt, s2_filt, mask = _validate_and_filter_arrays(e, sigma2)

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool

    def test_accepts_lists(self):
        """Should accept list inputs."""
        e = [1.0, 2.0, 3.0]
        sigma2 = [0.1, 0.2, 0.3]

        e_filt, s2_filt, mask = _validate_and_filter_arrays(e, sigma2)

        assert len(e_filt) == 3
        assert len(s2_filt) == 3


class TestClipProbability:
    """Test cases for _clip_probability function."""

    def test_clips_to_eps(self):
        """Should clip low probability to eps."""
        eps = 1e-10

        result = _clip_probability(0.0, eps)

        assert result == eps

    def test_clips_to_one_minus_eps(self):
        """Should clip high probability to 1-eps."""
        eps = 1e-10

        result = _clip_probability(1.0, eps)

        assert result == 1.0 - eps

    def test_leaves_valid_probability(self):
        """Should leave valid probabilities unchanged."""
        eps = 1e-10

        result = _clip_probability(0.5, eps)

        assert result == 0.5


class TestParseAlphas:
    """Test cases for parse_alphas function."""

    def test_parses_comma_separated(self):
        """Should parse comma-separated values."""
        result = parse_alphas("0.01,0.05,0.10")

        assert result == [0.01, 0.05, 0.10]

    def test_single_value(self):
        """Should handle single value."""
        result = parse_alphas("0.05")

        assert result == [0.05]

    def test_empty_string(self):
        """Should handle empty string."""
        result = parse_alphas("")

        assert result == []


class TestToNumpy:
    """Test cases for to_numpy function."""

    def test_converts_list(self):
        """Should convert list to numpy array."""
        lst = [1.0, 2.0, 3.0]

        result = to_numpy(lst)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_converts_numpy_array(self):
        """Should handle numpy array input."""
        arr = np.array([1.0, 2.0, 3.0])

        result = to_numpy(arr)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    def test_converts_to_float(self):
        """Should convert to float dtype."""
        lst = [1, 2, 3]

        result = to_numpy(lst)

        assert result.dtype == float


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
