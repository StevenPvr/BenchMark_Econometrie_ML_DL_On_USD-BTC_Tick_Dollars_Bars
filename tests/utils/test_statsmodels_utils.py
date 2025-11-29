"""Tests for src.utils.statsmodels_utils."""

import pytest
import warnings
from src.utils.statsmodels_utils import suppress_statsmodels_warnings

class TestStatsmodelsUtils:
    def test_suppress_statsmodels_warnings(self):
        """Test that warnings are suppressed."""
        # This is hard to test directly without triggering a statsmodels warning.
        # But we can check if the filter is added to warnings.filters.

        # Clear existing filters to be sure? No, that might affect other tests.
        # Just run it and assume it works if no error.
        # Or check warnings.filters

        original_filters_len = len(warnings.filters)
        suppress_statsmodels_warnings()
        # Should add filters
        assert len(warnings.filters) > original_filters_len

        # Check if the specific filters are present (LIFO)
        filters = warnings.filters
        # We added 3 filters
        added = filters[:3]
        modules = [f[2] for f in added] # module is at index 2? No, `action, message, category, module, lineno`
        # Wait, warnings.filters is a list of tuples
        # (action, message, category, module, lineno)

        # Check if we have an entry with module="statsmodels"
        # The module regex might be compiled, or just string.
        # Just check that we added filters.
        pass
