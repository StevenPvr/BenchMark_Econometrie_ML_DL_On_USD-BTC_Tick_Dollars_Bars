"""Tests for default labeling optimization ranges."""

from __future__ import annotations

import numpy as np

from src.label_primaire.optimize import LabelingHyperparams, compute_class_weights


def test_min_ret_range_matches_raw_log_return_units() -> None:
    """min_ret_range should use raw log-return units (returns not x100)."""
    params = LabelingHyperparams()

    assert params.min_ret_range == (0.00005, 0.002)


def test_compute_class_weights_balances_minor_class() -> None:
    """Balanced weights should upweight the rare class."""
    y = np.array([0] * 90 + [1] * 10)
    weights = compute_class_weights(y)

    assert weights[1] > weights[0]
