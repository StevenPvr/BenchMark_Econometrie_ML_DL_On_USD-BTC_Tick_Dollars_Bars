from __future__ import annotations

import sys
from pathlib import Path

import pytest
import pandas as pd
import numpy as np
from src.features.entropy import (
    compute_shannon_entropy,
    compute_approximate_entropy,
    compute_sample_entropy,
    _compute_shannon_entropy,
    _discretize_returns,
    _rolling_shannon_entropy,
    _count_similar_patterns,
    _compute_apen_single,
    _rolling_apen,
    _count_matches_sampen,
    _compute_sampen_single,
    _rolling_sampen,
)

# =============================================================================
# UNIT TESTS FOR NUMBA FUNCTIONS
# =============================================================================

def test_compute_shannon_entropy_numba():
    # counts=[10], n=10 -> p=1.0 -> H=0
    assert _compute_shannon_entropy(np.array([10], dtype=np.int64), 10) == 0.0

    # counts=[5, 5], n=10 -> p=0.5, 0.5 -> H = -0.5*ln(0.5) - 0.5*ln(0.5) = ln(2) ~= 0.693
    res = _compute_shannon_entropy(np.array([5, 5], dtype=np.int64), 10)
    assert np.isclose(res, np.log(2))

    # n=0
    assert np.isnan(_compute_shannon_entropy(np.array([], dtype=np.int64), 0))

def test_discretize_returns_numba():
    rets = np.array([-0.01, 0.0, 0.01], dtype=np.float64)
    bins = _discretize_returns(rets, n_bins=3)
    # min=-0.01, max=0.01, range=0.02, bin_width=0.00666
    # -0.01 -> bin 0
    # 0.0 -> bin 1
    # 0.01 -> bin 2
    assert np.all(bins == np.array([0, 1, 2]))

    # All same values
    rets_same = np.array([0.01, 0.01], dtype=np.float64)
    bins_same = _discretize_returns(rets_same, n_bins=3)
    assert np.all(bins_same == 0)

    # Empty/NaN
    assert len(_discretize_returns(np.array([], dtype=np.float64), 3)) == 0

def test_count_similar_patterns_numba():
    # Data: 1, 2, 1, 2
    # m=2, r=0.1
    # Patterns: [1,2], [2,1], [1,2]
    # P0=[1,2] matches P2=[1,2]
    data = np.array([1., 2., 1., 2.], dtype=np.float64)
    # n=4, m=2 -> n_patterns = 3
    # P0 matches P0, P2 -> count=2
    # P1 matches P1 -> count=1
    # P2 matches P0, P2 -> count=2
    # Sum log(count/3): log(2/3) + log(1/3) + log(2/3)
    expected = (np.log(2/3) + np.log(1/3) + np.log(2/3)) / 3
    res = _count_similar_patterns(data, m=2, r=0.1)
    assert np.isclose(res, expected)

    # Edge case: n < m
    assert np.isnan(_count_similar_patterns(data, m=5, r=0.1))

def test_count_matches_sampen_numba():
    # Data: 1, 2, 1, 2, 1
    # m=2, r=0.1
    # n=5, n_patterns = 3
    # Patterns (m): [1,2] (i=0), [2,1] (i=1), [1,2] (i=2)
    # Pairs (i, j): (0,1), (0,2), (1,2)
    # (0,1): [1,2] vs [2,1] -> distinct.
    # (0,2): [1,2] vs [1,2] -> match (B+=1). Check m+1: [1,2,1] vs [1,2,1] -> match (A+=1)
    # (1,2): [2,1] vs [1,2] -> distinct.

    data = np.array([1., 2., 1., 2., 1.], dtype=np.float64)
    B, A = _count_matches_sampen(data, m=2, r=0.1)
    assert B == 1
    assert A == 1

    # Edge case: n < m+1
    assert _count_matches_sampen(data[:2], m=2, r=0.1) == (0, 0)

def test_rolling_functions_short_input():
    window = 5
    data = np.array([1., 2., 3.], dtype=np.float64)

    # Shannon
    res_sh = _rolling_shannon_entropy(data, window, 2)
    assert np.all(np.isnan(res_sh))

    # ApEn
    res_ap = _rolling_apen(data, window, 2, 0.2)
    assert np.all(np.isnan(res_ap))

    # SampEn
    res_sa = _rolling_sampen(data, window, 2, 0.2)
    assert np.all(np.isnan(res_sa))

def test_rolling_functions_constant_input():
    # Constant input -> std=0 -> entropy=0 or handled
    window = 3
    data = np.array([1., 1., 1., 1., 1.], dtype=np.float64)

    # Shannon: min=max -> bins=0 -> counts=[N, 0..] -> H=0
    res_sh = _rolling_shannon_entropy(data, window, 2)
    # For constant input, max_val == min_val, so bins are all 0.
    # Count[0] = window. p=1. H=0.
    assert res_sh[-1] == 0.0

    # ApEn: std < epsilon -> 0
    res_ap = _rolling_apen(data, window, 2, 0.2)
    assert res_ap[-1] == 0.0

    # SampEn: std < epsilon -> 0
    res_sa = _rolling_sampen(data, window, 2, 0.2)
    assert res_sa[-1] == 0.0

# =============================================================================
# INTEGRATION TESTS
# =============================================================================

@pytest.fixture
def test_df():
    # Create a deterministic pattern
    # 0, 1, 0, 1... highly predictable
    return pd.DataFrame({
        "log_return": [0.01, -0.01] * 50
    })

def test_compute_shannon_entropy(test_df):
    df = compute_shannon_entropy(test_df, window=10, n_bins=2)
    assert isinstance(df, pd.Series)
    assert "shannon_entropy_10" == df.name
    # Alternating returns: -0.01, 0.01. Two bins.
    # Ideally split evenly -> max entropy ln(2) ~= 0.69
    # Depending on binning edges, it might put all in one or split.
    # With 2 bins, min=-0.01, max=0.01. Mid=0.
    # -0.01 -> bin 0. 0.01 -> bin 1.
    # So we expect high entropy.
    valid = df.dropna()
    assert len(valid) > 0
    assert valid.iloc[-1] > 0

def test_compute_approximate_entropy(test_df):
    s = compute_approximate_entropy(test_df, window=10, m=2)
    assert isinstance(s, pd.Series)
    valid = s.dropna()
    assert len(valid) > 0
    # Alternating 0,1,0,1 pattern is very regular -> low entropy
    # But we normalize by std.
    # Pattern [0,1] repeats.
    # ApEn should be relatively low (compared to random noise)
    assert s.name == "apen_10"

def test_compute_sample_entropy(test_df):
    s = compute_sample_entropy(test_df, window=10, m=2)
    assert isinstance(s, pd.Series)
    valid = s.dropna()
    assert len(valid) > 0

def test_compute_sampen_single_edge_cases():
    # Case where B=0
    # If no matches of length m found, it returns 5.0 (cap)
    # To force B=0, we need patterns that are all very different (distance > r)
    # r is after normalization (std=1), so r=0.2
    # e.g. 0, 10, 0, 10 (normalized: -1, 1, -1, 1). Distance=2 > 0.2.
    # Wait, 0,10,0,10... m=2. [0,10] vs [0,10]. Distance 0.
    # We need a sequence with NO repeated patterns.
    # [0, 10, 20, 30, 40]
    data = np.array([0, 10, 20, 30, 40], dtype=np.float64)
    # Normalize manually approx
    data = (data - np.mean(data)) / np.std(data)

    # New signature includes max_entropy parameter (adaptive based on window size)
    max_entropy = np.log(len(data))  # Adaptive max = log(n)
    val = _compute_sampen_single(data, m=2, r=0.01, max_entropy=max_entropy)
    assert val == max_entropy  # Adaptive cap based on window size

    # Case where B>0 but A=0
    # Matches of length m exist, but not m+1
    # [0, 1, 100]
    # [0, 1, -100]
    # m=2. [0,1] matches [0,1].
    # m+1. [0,1,100] vs [0,1,-100]. Diff=200 > r.
    data = np.array([0, 1, 100, 0, 1, -100], dtype=np.float64)
    # normalize
    data = (data - np.mean(data)) / np.std(data)

    max_entropy = np.log(len(data))
    val = _compute_sampen_single(data, m=2, r=0.1, max_entropy=max_entropy)
    # Should be high (capped at adaptive max)
    assert val == max_entropy
