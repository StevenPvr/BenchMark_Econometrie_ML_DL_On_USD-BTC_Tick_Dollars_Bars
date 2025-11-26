"""Tests for feature engineering helpers."""

from __future__ import annotations

import os
import sys

import pandas as pd  # type: ignore[import-untyped]

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.features.main import shift_target_to_future_return


def test_shift_target_to_future_return_aligns_next_bar():
    """Target should be shifted to the next step and last row dropped."""
    df = pd.DataFrame(
        {
            "log_return": [0.1, 0.2, 0.3],
            "feature_a": [1.0, 2.0, 3.0],
        }
    )

    shifted = shift_target_to_future_return(df, target_col="log_return")

    # After shift(-1), we expect two rows with next-bar returns
    assert shifted["log_return"].tolist() == [0.2, 0.3]
    # Other features remain aligned with original rows
    assert shifted["feature_a"].tolist() == [1.0, 2.0]
