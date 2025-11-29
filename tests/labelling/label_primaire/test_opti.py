"""
Unit tests for label_primaire/opti.py

Tests the optimization module, including Walk-Forward CV and Optuna integration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch, ANY

from src.labelling.label_primaire.opti import (
    WalkForwardCV,
    optimize_model,
    OptimizationConfig,
    OptimizationResult,
    _generate_trial_events,
    _validate_events,
    _align_features_events,
    _run_cv_scoring,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create sample features and events for testing."""
    n_samples = 1000
    dates = pd.date_range("2024-01-01", periods=n_samples, freq="h")

    X = pd.DataFrame({
        "feature_1": np.random.randn(n_samples),
        "feature_2": np.random.randn(n_samples),
    }, index=dates)

    events = pd.DataFrame({
        "label": np.random.choice([-1, 0, 1], size=n_samples),
        "t1": dates + pd.Timedelta(hours=2), # short holding period
    }, index=dates)

    return X, events


@pytest.fixture
def mock_dollar_bars() -> pd.DataFrame:
    """Create mock dollar bars."""
    dates = pd.date_range("2024-01-01", periods=100, freq="h")
    df = pd.DataFrame({
        "close": [100.0 + i for i in range(100)],
        "datetime_close": dates,
    })
    return df.set_index("datetime_close")


# =============================================================================
# TESTS - WalkForwardCV
# =============================================================================


class TestWalkForwardCV:
    """Tests for WalkForwardCV class."""

    def test_split_indices(self, sample_data):
        """Test that splits return valid indices."""
        X, events = sample_data
        cv = WalkForwardCV(n_splits=3, min_train_size=100, embargo_pct=0.01)

        splits = cv.split(X, events)

        assert len(splits) > 0

        for train_idx, val_idx in splits:
            # Check indices are within bounds
            assert train_idx.min() >= 0
            assert train_idx.max() < len(X)
            assert val_idx.min() >= 0
            assert val_idx.max() < len(X)

            # Check disjoint (no overlap)
            # Actually with purging they shouldn't overlap in time, but indices might
            # be disjoint by definition of TimeSeriesSplit.
            # TimeSeriesSplit: train indices are always before val indices.
            assert train_idx.max() < val_idx.min()

            # Embargo check: gap between train and val
            embargo_size = int(len(X) * 0.01)
            gap = val_idx.min() - train_idx.max()
            assert gap >= embargo_size

    def test_purging(self, sample_data):
        """Test that purging removes overlapping samples."""
        X, events = sample_data

        # Create an event that overlaps with validation
        # Set t1 of the last training sample to be inside validation period

        # Manually construct a split scenario
        # train: 0-100, val: 110-150 (embargo 10)
        # If event at 99 ends at 120, it should be purged.

        cv = WalkForwardCV(n_splits=2, min_train_size=50, embargo_pct=0.0)

        # Mock split to return fixed indices for testing internal logic
        # But split() uses TimeSeriesSplit internally.
        # Let's test _apply_purging directly.

        train_idx = np.arange(100)
        val_idx = np.arange(100, 150)

        # Case 1: No overlap
        events.iloc[99, events.columns.get_loc("t1")] = X.index[99] # ends immediately
        purged_idx = cv._apply_purging(train_idx, val_idx, X, events)
        assert 99 in purged_idx

        # Case 2: Overlap
        val_start_time = X.index[100]
        # set t1 of index 99 to be after val_start
        events.iloc[99, events.columns.get_loc("t1")] = val_start_time + pd.Timedelta(seconds=1)

        purged_idx = cv._apply_purging(train_idx, val_idx, X, events)
        assert 99 not in purged_idx


# =============================================================================
# TESTS - Optimization Helpers
# =============================================================================


class TestOptiHelpers:
    """Tests for helper functions in opti.py."""

    def test_validate_events(self):
        """Test event validation."""
        config = OptimizationConfig(model_name="test", min_train_size=10)

        # Valid events
        events = pd.DataFrame({
            "label": [1, -1] * 10
        })
        valid, reason = _validate_events(events, config)
        assert valid
        assert reason == "OK"

        # Empty events
        events_empty = pd.DataFrame()
        valid, reason = _validate_events(events_empty, config)
        assert not valid
        assert "empty" in reason

        # Not enough events
        events_small = pd.DataFrame({"label": [1, -1]})
        valid, reason = _validate_events(events_small, config)
        assert not valid
        assert "not enough" in reason

        # Only one class
        events_one_class = pd.DataFrame({"label": [1] * 20})
        valid, reason = _validate_events(events_one_class, config)
        assert not valid
        assert "only 1 class" in reason

    def test_align_features_events(self):
        """Test feature-event alignment."""
        config = OptimizationConfig(model_name="test", min_train_size=5)

        dates = pd.date_range("2024-01-01", periods=10, freq="h")
        X = pd.DataFrame({"f1": range(10)}, index=dates)
        events = pd.DataFrame({"label": [1]*10}, index=dates)

        # Perfect alignment
        X_out, y_out, ev_out, reason = _align_features_events(X, events, config)
        assert X_out is not None
        assert len(X_out) == 10

        # Misalignment
        events_mis = events.iloc[5:]
        X_out, y_out, ev_out, reason = _align_features_events(X, events_mis, config)
        assert len(X_out) == 5

        # Too few common
        events_tiny = events.iloc[:2]
        X_out, y_out, ev_out, reason = _align_features_events(X, events_tiny, config)
        assert X_out is None
        assert "not enough" in reason


# =============================================================================
# TESTS - Full Optimization Flow (Mocked)
# =============================================================================


@patch("src.labelling.label_primaire.opti.load_model_class")
@patch("src.labelling.label_primaire.opti.get_dataset_for_model")
@patch("src.labelling.label_primaire.opti.load_dollar_bars")
@patch("src.labelling.label_primaire.opti.get_daily_volatility")
@patch("src.labelling.label_primaire.opti.optuna.create_study")
def test_optimize_model(
    mock_create_study,
    mock_get_vol,
    mock_load_bars,
    mock_get_dataset,
    mock_load_model_class,
    mock_dollar_bars,
    sample_data,
    tmp_path,
):
    """Test optimize_model function."""

    # Setup mocks
    X, events = sample_data
    mock_get_dataset.return_value = X
    mock_load_bars.return_value = mock_dollar_bars
    mock_get_vol.return_value = pd.Series(0.01, index=mock_dollar_bars.index)

    # Mock model class
    mock_model_class = MagicMock()
    mock_load_model_class.return_value = mock_model_class

    # Mock Optuna study
    mock_study = MagicMock()
    mock_create_study.return_value = mock_study

    # Mock best trial
    mock_trial = MagicMock()
    mock_trial.params = {
        "pt_mult": 1.0,
        "sl_mult": 1.0,
        "max_holding": 10,
        "min_return": 0.0,
        "n_estimators": 100, # Model param
    }
    mock_trial.value = 0.6
    mock_study.best_trial = mock_trial
    mock_study.trials = [mock_trial]

    # Run optimization
    with patch("src.labelling.label_primaire.opti.LABEL_PRIMAIRE_OPTI_DIR", tmp_path):
        result = optimize_model(
            model_name="lightgbm",
            config=OptimizationConfig(
                model_name="lightgbm",
                n_trials=1,
                n_splits=2,
                min_train_size=10,
                data_fraction=1.0, # Use all data
            )
        )

    # Verification
    assert isinstance(result, OptimizationResult)
    assert result.model_name == "lightgbm"
    assert result.best_score == 0.6

    # Check that optimization was run
    mock_study.optimize.assert_called_once()

    # Check results file
    assert (tmp_path / "lightgbm_optimization.json").exists()
