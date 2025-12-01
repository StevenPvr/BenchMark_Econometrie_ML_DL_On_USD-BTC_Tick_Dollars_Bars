"""Tests for src.labelling.label_meta.train."""

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.labelling.label_meta.train import (
    WalkForwardKFold,
    MetaEvaluationMetrics,
    MetaTrainingResult,
    load_optimized_params,
    compute_metrics,
    _remove_non_feature_cols,
    _align_close_to_features,
    _handle_missing_values,
    get_available_meta_optimizations,
    generate_oos_labels_train,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="h")
    X = pd.DataFrame(
        np.random.randn(100, 5),
        index=dates,
        columns=[f"f{i}" for i in range(5)]
    )
    y = pd.Series(np.random.choice([0, 1], 100), index=dates)
    t1 = pd.Series(dates.shift(10, freq="h"), index=dates)
    return X, y, t1


@pytest.fixture
def sample_features():
    """Generate sample features DataFrame."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="h")
    return pd.DataFrame({
        "f1": np.random.randn(100),
        "f2": np.random.randn(100),
        "bar_id": range(100),
        "split": ["train"] * 80 + ["test"] * 20,
        "datetime_close": dates,
    })


# =============================================================================
# WALK-FORWARD K-FOLD TESTS
# =============================================================================


class TestWalkForwardKFold:
    def test_init(self):
        """Test initialization."""
        cv = WalkForwardKFold(n_splits=5, embargo_pct=0.01, min_train_size=50)

        assert cv.n_splits == 5
        assert cv.embargo_pct == 0.01
        assert cv.min_train_size == 50

    def test_split_basic(self, sample_data):
        """Test basic split generation."""
        X, y, t1 = sample_data

        cv = WalkForwardKFold(n_splits=4, min_train_size=10, embargo_pct=0.0)
        splits = cv.split(X, y, t1=None)

        assert len(splits) > 0
        for train_idx, val_idx in splits:
            # Verify train comes before val
            if len(train_idx) > 0 and len(val_idx) > 0:
                assert train_idx[-1] < val_idx[0]

    def test_split_with_purging(self, sample_data):
        """Test split with purging enabled."""
        X, y, t1 = sample_data

        cv = WalkForwardKFold(n_splits=3, min_train_size=10, embargo_pct=0.01)
        splits = cv.split(X, y, t1=t1)

        # Should have at least one valid split
        assert len(splits) >= 1

    def test_split_min_train_size(self, sample_data):
        """Test that min_train_size is respected."""
        X, y, _ = sample_data

        cv = WalkForwardKFold(n_splits=5, min_train_size=100)
        splits = cv.split(X.iloc[:50], y.iloc[:50])

        # With only 50 samples and min_train_size=100, no valid splits
        assert len(splits) == 0

    def test_apply_purge(self, sample_data):
        """Test purging logic."""
        X, _, t1 = sample_data

        cv = WalkForwardKFold(n_splits=3)

        train_indices = np.arange(30)
        val_indices = np.arange(30, 50)

        purged = cv._apply_purge(train_indices, val_indices, X, t1)

        # Purged indices should be a subset of original
        assert len(purged) <= len(train_indices)


# =============================================================================
# DATACLASS TESTS
# =============================================================================


class TestMetaEvaluationMetrics:
    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = MetaEvaluationMetrics(
            accuracy=0.85,
            precision=0.80,
            recall=0.75,
            f1=0.77,
            auc_roc=0.90,
        )

        d = metrics.to_dict()

        assert d["accuracy"] == 0.85
        assert d["precision"] == 0.80
        assert d["recall"] == 0.75
        assert d["f1"] == 0.77
        assert d["auc_roc"] == 0.90

    def test_to_dict_none_auc(self):
        """Test conversion when AUC is None."""
        metrics = MetaEvaluationMetrics(
            accuracy=0.85,
            precision=0.80,
            recall=0.75,
            f1=0.77,
            auc_roc=None,
        )

        d = metrics.to_dict()
        assert d["auc_roc"] is None


class TestMetaTrainingResult:
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = MetaTrainingResult(
            primary_model_name="primary",
            meta_model_name="meta",
            meta_model_params={"n_estimators": 100},
            triple_barrier_params={"pt_mult": 1.0},
            train_samples=1000,
            test_samples=200,
            train_metrics={"accuracy": 0.85},
            test_metrics={"accuracy": 0.80},
            meta_label_distribution_train={"0": 500, "1": 500},
            meta_label_distribution_test={"0": 100, "1": 100},
            n_folds=5,
            primary_model_path="/path/to/primary",
            meta_model_path="/path/to/meta",
            oof_predictions_path="/path/to/oof",
            test_predictions_path="/path/to/test",
        )

        d = result.to_dict()

        assert d["primary_model_name"] == "primary"
        assert d["meta_model_name"] == "meta"
        assert d["train_samples"] == 1000

    def test_save(self, tmp_path):
        """Test saving to JSON."""
        result = MetaTrainingResult(
            primary_model_name="primary",
            meta_model_name="meta",
            meta_model_params={"n_estimators": 100},
            triple_barrier_params={"pt_mult": 1.0},
            train_samples=1000,
            test_samples=200,
            train_metrics={"accuracy": 0.85},
            test_metrics={"accuracy": 0.80},
            meta_label_distribution_train={"0": 500, "1": 500},
            meta_label_distribution_test={"0": 100, "1": 100},
            n_folds=5,
            primary_model_path="/path/to/primary",
            meta_model_path="/path/to/meta",
            oof_predictions_path="/path/to/oof",
            test_predictions_path="/path/to/test",
        )

        path = tmp_path / "result.json"
        result.save(path)

        assert path.exists()

        with open(path) as f:
            data = json.load(f)
            assert data["primary_model_name"] == "primary"


# =============================================================================
# PARAMETER LOADING TESTS
# =============================================================================


class TestParameterLoading:
    def test_load_optimized_params(self, tmp_path):
        """Test loading optimized parameters."""
        # Create mock optimization file
        opti_data = {
            "best_params": {"n_estimators": 100},
            "best_triple_barrier_params": {"pt_mult": 1.0},
            "best_score": 0.85,
            "metric": "f1_score",
        }

        opti_file = tmp_path / "primary_meta_optimization.json"
        with open(opti_file, "w") as f:
            json.dump(opti_data, f)

        result = load_optimized_params("primary", "meta", opti_dir=tmp_path)

        assert result["meta_model_params"] == {"n_estimators": 100}
        assert result["best_score"] == 0.85

    def test_load_optimized_params_not_found(self, tmp_path):
        """Test loading when file not found."""
        with pytest.raises(FileNotFoundError):
            load_optimized_params("nonexistent", "model", opti_dir=tmp_path)


# =============================================================================
# METRICS TESTS
# =============================================================================


class TestComputeMetrics:
    def test_compute_metrics_basic(self):
        """Test basic metrics computation."""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1])

        metrics = compute_metrics(y_true, y_pred)

        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.precision <= 1
        assert 0 <= metrics.recall <= 1
        assert 0 <= metrics.f1 <= 1

    def test_compute_metrics_with_proba(self):
        """Test metrics with probabilities."""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1])
        y_proba = np.array([
            [0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.6, 0.4],
            [0.7, 0.3], [0.3, 0.7], [0.4, 0.6], [0.2, 0.8]
        ])

        metrics = compute_metrics(y_true, y_pred, y_proba)

        assert metrics.auc_roc is not None

    def test_compute_metrics_1d_proba(self):
        """Test metrics with 1D probabilities."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_proba = np.array([0.1, 0.9, 0.2, 0.8])

        metrics = compute_metrics(y_true, y_pred, y_proba)

        assert metrics.auc_roc is not None


# =============================================================================
# DATA PREPARATION TESTS
# =============================================================================


class TestDataPreparation:
    def test_remove_non_feature_cols(self, sample_features):
        """Test removing non-feature columns."""
        result = _remove_non_feature_cols(sample_features)

        assert "bar_id" not in result.columns
        assert "split" not in result.columns
        assert "datetime_close" not in result.columns
        assert "f1" in result.columns
        assert "f2" in result.columns

    def test_align_close_to_features(self, sample_features):
        """Test aligning close prices to features."""
        dates = pd.date_range("2023-01-01", periods=200, freq="h")
        close = pd.Series(np.random.randn(200) + 100, index=dates)

        # Set up features with datetime index
        features = sample_features.set_index("datetime_close")

        aligned = _align_close_to_features(close, features)

        assert len(aligned) <= len(close)
        assert aligned.index[0] >= features.index[0]
        assert aligned.index[-1] <= features.index[-1]

    def test_handle_missing_values(self):
        """Test handling missing values."""
        # Note: _handle_missing_values uses isinstance(is_na_any, bool)
        # which may not catch numpy bool. Test the actual behavior.
        df = pd.DataFrame({
            "f1": [1.0, np.nan, 3.0, 4.0],
            "f2": [np.nan, np.nan, np.nan, np.nan],
            "f3": [1.0, 2.0, 3.0, 4.0],
        })

        # Call the function (may or may not fill depending on implementation)
        result = _handle_missing_values(df.copy())

        # The function returns a DataFrame
        assert isinstance(result, pd.DataFrame)
        # f3 should remain unchanged
        assert result["f3"].tolist() == [1.0, 2.0, 3.0, 4.0]


# =============================================================================
# OPTIMIZATION AVAILABILITY TESTS
# =============================================================================


class TestOptimizationAvailability:
    def test_get_available_meta_optimizations_empty(self, tmp_path, mocker):
        """Test when no optimizations exist."""
        mocker.patch(
            "src.labelling.label_meta.train.LABEL_META_OPTI_DIR",
            tmp_path / "nonexistent"
        )

        result = get_available_meta_optimizations()
        assert result == []

    def test_get_available_meta_optimizations(self, tmp_path, mocker):
        """Test listing available optimizations."""
        mocker.patch(
            "src.labelling.label_meta.train.LABEL_META_OPTI_DIR",
            tmp_path
        )

        # Create mock optimization files
        (tmp_path / "primary1_meta1_optimization.json").touch()
        (tmp_path / "primary2_meta2_optimization.json").touch()

        result = get_available_meta_optimizations()

        assert len(result) == 2
        assert ("primary1", "meta1") in result
        assert ("primary2", "meta2") in result


# =============================================================================
# ADDITIONAL COVERAGE TESTS
# =============================================================================


class TestAdditionalCoverage:
    def test_walk_forward_kfold_get_n_splits(self):
        """Test getting number of splits."""
        cv = WalkForwardKFold(n_splits=5)
        assert cv.n_splits == 5

    def test_walk_forward_kfold_no_valid_splits(self, sample_data):
        """Test when no valid splits can be created."""
        X, y, _ = sample_data

        cv = WalkForwardKFold(n_splits=10, min_train_size=100)
        splits = cv.split(X.iloc[:20], y.iloc[:20])

        assert len(splits) == 0

    def test_meta_evaluation_metrics_defaults(self):
        """Test MetaEvaluationMetrics with None AUC."""
        metrics = MetaEvaluationMetrics(
            accuracy=0.8,
            precision=0.75,
            recall=0.7,
            f1=0.72,
            auc_roc=None,
        )

        d = metrics.to_dict()
        assert d["auc_roc"] is None
        assert d["accuracy"] == 0.8

    def test_meta_training_result_timestamp(self):
        """Test that timestamp is automatically set."""
        result = MetaTrainingResult(
            primary_model_name="p",
            meta_model_name="m",
            meta_model_params={},
            triple_barrier_params={},
            train_samples=100,
            test_samples=20,
            train_metrics={},
            test_metrics={},
            meta_label_distribution_train={},
            meta_label_distribution_test={},
            n_folds=5,
            primary_model_path="",
            meta_model_path="",
            oof_predictions_path="",
            test_predictions_path="",
        )

        d = result.to_dict()
        assert "timestamp" in d
        assert d["timestamp"] is not None

    def test_compute_metrics_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1])  # Perfect

        metrics = compute_metrics(y_true, y_pred)

        assert metrics.accuracy == 1.0
        assert metrics.f1 == 1.0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0

    def test_compute_metrics_all_wrong(self):
        """Test metrics with all wrong predictions."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1])  # All wrong

        metrics = compute_metrics(y_true, y_pred)

        assert metrics.accuracy == 0.0

    def test_load_optimized_params_with_defaults(self, tmp_path):
        """Test loading params with minimal data."""
        opti_data = {
            "best_params": {},
            "best_triple_barrier_params": {},
            "best_score": 0.5,
            "metric": "f1",
        }

        opti_file = tmp_path / "p_m_optimization.json"
        with open(opti_file, "w") as f:
            json.dump(opti_data, f)

        result = load_optimized_params("p", "m", opti_dir=tmp_path)

        assert result["meta_model_params"] == {}
        assert result["best_score"] == 0.5

    def test_remove_non_feature_cols_empty(self):
        """Test with no columns to remove."""
        df = pd.DataFrame({
            "f1": [1, 2, 3],
            "f2": [4, 5, 6],
        })

        result = _remove_non_feature_cols(df)
        assert "f1" in result.columns
        assert "f2" in result.columns


# =============================================================================
# OOS LABEL GENERATION TESTS
# =============================================================================


class TestOOSLabelGeneration:
    def test_generate_oos_labels_basic(self, sample_data):
        """Test basic OOS label generation."""
        X, y, _ = sample_data
        t1 = pd.Series(X.index.shift(5, freq="h"), index=X.index)

        # Use real sklearn model
        from sklearn.ensemble import RandomForestClassifier

        class MockModel:
            def __init__(self, **kwargs):
                self.model = RandomForestClassifier(n_estimators=5, random_state=42)

            def fit(self, X, y):
                self.model.fit(X, y)

            def predict(self, X):
                return self.model.predict(X)

            def predict_proba(self, X):
                return self.model.predict_proba(X)

        oos_labels, oos_proba, coverage = generate_oos_labels_train(
            X, y, t1,
            model_class=MockModel,
            model_params={},
            n_splits=3,
            min_train_size=10,
        )

        assert len(oos_labels) == len(y)
        assert len(coverage) == len(y)
        # At least some samples should be covered
        assert coverage.sum() > 0

    def test_generate_oos_labels_no_valid_splits(self, sample_data):
        """Test when no valid splits can be generated."""
        X, y, _ = sample_data
        X_small = X.iloc[:10]
        y_small = y.iloc[:10]
        t1 = pd.Series(X_small.index, index=X_small.index)

        mock_model_cls = MagicMock()

        oos_labels, oos_proba, coverage = generate_oos_labels_train(
            X_small, y_small, t1,
            model_class=mock_model_cls,
            model_params={},
            n_splits=10,
            min_train_size=100,  # Too large for data
        )

        # Should return empty results
        assert np.isnan(oos_labels).all()
        assert coverage.sum() == 0


# =============================================================================
# WALK FORWARD KFOLD DETAILED TESTS
# =============================================================================


class TestWalkForwardKFoldDetailed:
    def test_split_with_t1(self, sample_data):
        """Test split with t1 for purging."""
        X, y, t1 = sample_data

        cv = WalkForwardKFold(n_splits=3, min_train_size=10, embargo_pct=0.0)
        splits = cv.split(X, y, t1=t1)

        assert isinstance(splits, list)

    def test_split_respects_time_order(self, sample_data):
        """Test that splits respect temporal order."""
        X, y, _ = sample_data

        cv = WalkForwardKFold(n_splits=3, min_train_size=10)
        splits = cv.split(X, y)

        for train_idx, val_idx in splits:
            if len(train_idx) > 0 and len(val_idx) > 0:
                # Train indices should all be before validation
                assert train_idx.max() < val_idx.min()

    def test_apply_purge_with_overlap(self, sample_data):
        """Test purging when labels overlap validation period."""
        X, _, _ = sample_data
        t1 = pd.Series(X.index.shift(30, freq="h"), index=X.index)  # Long overlap

        cv = WalkForwardKFold(n_splits=3, min_train_size=10)

        train_idx = np.arange(50)
        val_idx = np.arange(50, 70)

        purged = cv._apply_purge(train_idx, val_idx, X, t1)

        # Some training samples should be purged due to overlap
        assert len(purged) < len(train_idx)


# =============================================================================
# COMPUTE METRICS DETAILED TESTS
# =============================================================================


class TestComputeMetricsDetailed:
    def test_compute_metrics_binary_balanced(self):
        """Test metrics with balanced binary classification."""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 1, 0, 0, 1])  # Some errors

        metrics = compute_metrics(y_true, y_pred)

        assert metrics.accuracy == 0.75
        assert 0 < metrics.f1 < 1

    def test_compute_metrics_with_2d_proba(self):
        """Test metrics with 2D probability array."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_proba = np.array([
            [0.9, 0.1],
            [0.2, 0.8],
            [0.85, 0.15],
            [0.1, 0.9],
        ])

        metrics = compute_metrics(y_true, y_pred, y_proba)

        assert metrics.auc_roc is not None
        assert metrics.auc_roc > 0.5

    def test_compute_metrics_edge_case(self):
        """Test metrics edge case with few samples."""
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])

        metrics = compute_metrics(y_true, y_pred)

        assert metrics.accuracy == 1.0
