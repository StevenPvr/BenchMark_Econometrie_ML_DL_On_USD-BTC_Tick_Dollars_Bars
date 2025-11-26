"""Tests for src/garch/garch_params/data.py module."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.garch.garch_params.data import (
    _validate_residuals,
)
from src.constants import GARCH_ESTIMATION_MIN_OBSERVATIONS


class TestValidateResiduals:
    """Test cases for _validate_residuals function."""

    def test_passes_for_sufficient_data(self):
        """Should not raise when sufficient observations."""
        # Create array with more than minimum observations
        resid_train = np.random.randn(GARCH_ESTIMATION_MIN_OBSERVATIONS + 100)

        # Should not raise
        _validate_residuals(resid_train)

    def test_raises_for_insufficient_data(self):
        """Should raise ValueError for insufficient observations."""
        # Create array with fewer than minimum observations
        resid_train = np.random.randn(GARCH_ESTIMATION_MIN_OBSERVATIONS - 1)

        with pytest.raises(ValueError, match="Insufficient training residuals"):
            _validate_residuals(resid_train)

    def test_raises_for_empty_array(self):
        """Should raise ValueError for empty array."""
        resid_train = np.array([])

        with pytest.raises(ValueError, match="Insufficient training residuals"):
            _validate_residuals(resid_train)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
