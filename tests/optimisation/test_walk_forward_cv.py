"""Tests for walk-forward cross-validation logic."""

import numpy as np
import pytest

from src.optimisation.walk_forward_cv import (
    WalkForwardConfig,
    WalkForwardSplitter,
)


def test_walk_forward_splitter_basic():
    """Test basic splitting logic."""
    n_samples = 100
    n_splits = 3
    purge_gap = 2
    min_train_size = 20
    
    config = WalkForwardConfig(
        n_splits=n_splits,
        purge_gap=purge_gap,
        min_train_size=min_train_size,
        window_type="expanding"
    )
    
    splitter = WalkForwardSplitter(config)
    folds = list(splitter.split(n_samples))
    
    assert len(folds) == n_splits
    
    for fold in folds:
        # Check temporal order: train < test
        assert fold.train_end <= fold.test_start
        
        # Check purge gap
        assert fold.test_start - fold.train_end == purge_gap
        
        # Check indices validity
        assert len(fold.train_indices) > 0
        assert len(fold.test_indices) > 0
        assert fold.train_indices[-1] < fold.test_indices[0]
        
        # Check continuity
        assert fold.train_indices[-1] == fold.train_end - 1
        assert fold.test_indices[0] == fold.test_start


def test_walk_forward_splitter_expanding():
    """Test expanding window behavior."""
    n_samples = 100
    config = WalkForwardConfig(
        n_splits=3,
        purge_gap=0,
        min_train_size=20,
        window_type="expanding"
    )
    
    splitter = WalkForwardSplitter(config)
    folds = list(splitter.split(n_samples))
    
    # Train size should increase
    train_sizes = [len(f.train_indices) for f in folds]
    assert train_sizes == sorted(train_sizes)
    assert len(set(train_sizes)) == 3  # All different


def test_walk_forward_splitter_rolling():
    """Test rolling window behavior."""
    n_samples = 100
    rolling_size = 30
    config = WalkForwardConfig(
        n_splits=3,
        purge_gap=0,
        min_train_size=20,
        window_type="rolling",
        rolling_window_size=rolling_size
    )
    
    splitter = WalkForwardSplitter(config)
    folds = list(splitter.split(n_samples))
    
    # Train size should be capped at rolling_size (except possibly first if smaller)
    for fold in folds:
        assert len(fold.train_indices) <= rolling_size


def test_walk_forward_splitter_insufficient_data():
    """Test error handling for insufficient data."""
    n_samples = 50
    config = WalkForwardConfig(
        n_splits=5,
        purge_gap=10,  # Too large gap
        min_train_size=20
    )
    
    splitter = WalkForwardSplitter(config)
    with pytest.raises(ValueError, match="Need at least"):
        list(splitter.split(n_samples))


def test_walk_forward_splitter_purge_gap():
    """Explicitly verify purge gap logic."""
    n_samples = 50
    purge_gap = 5
    config = WalkForwardConfig(
        n_splits=2,
        purge_gap=purge_gap,
        min_train_size=10
    )
    
    splitter = WalkForwardSplitter(config)
    folds = list(splitter.split(n_samples))
    
    for fold in folds:
        # The gap between last train index and first test index should be purge_gap
        # train_end is exclusive, test_start is inclusive
        # so test_start - train_end is the gap size
        assert fold.test_start - fold.train_end == purge_gap
