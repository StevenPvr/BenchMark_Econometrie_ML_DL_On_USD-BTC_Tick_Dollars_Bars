"""
Unit tests for label_primaire/main.py

Tests the training pipeline with out-of-fold predictions using synthetic data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from src.labelling.label_primaire.main import (
    PurgedKFold,
    compute_metrics,
    generate_oof_predictions,
    EvaluationMetrics,
    TrainingResult,
)


# =============================================================================
# FIXTURES - Synthetic Data Generation
# =============================================================================


@pytest.fixture
def synthetic_features() -> pd.DataFrame:
    """Create synthetic feature DataFrame with datetime index."""
    n_samples = 1000
    np.random.seed(42)

    dates = pd.date_range("2024-01-01", periods=n_samples, freq="h")

    # Create features with some signal
    features = pd.DataFrame({
        "feature_1": np.random.randn(n_samples),
        "feature_2": np.random.randn(n_samples),
        "feature_3": np.random.randn(n_samples) * 0.5,
        "feature_4": np.cumsum(np.random.randn(n_samples) * 0.1),
        "feature_5": np.sin(np.arange(n_samples) * 0.1) + np.random.randn(n_samples) * 0.1,
    }, index=dates)

    return features


@pytest.fixture
def synthetic_labels(synthetic_features: pd.DataFrame) -> pd.Series:
    """Create synthetic labels correlated with features."""
    np.random.seed(42)
    n_samples = len(synthetic_features)

    # Create labels with some pattern (based on features)
    signal = (
        synthetic_features["feature_1"] * 0.3 +
        synthetic_features["feature_2"] * 0.2 +
        np.random.randn(n_samples) * 0.5
    )

    # Convert to classes: -1, 0, 1
    labels = pd.Series(
        np.where(signal > 0.3, 1, np.where(signal < -0.3, -1, 0)),
        index=synthetic_features.index,
        name="label"
    )

    return labels


@pytest.fixture
def synthetic_t1(synthetic_features: pd.DataFrame) -> pd.Series:
    """Create synthetic t1 (label end times)."""
    dates = synthetic_features.index
    # t1 is typically a few periods after the start
    t1_values = [dates[min(i + 5, len(dates) - 1)] for i in range(len(dates))]
    return pd.Series(t1_values, index=dates, name="t1")


# =============================================================================
# TESTS - PurgedKFold
# =============================================================================


class TestPurgedKFold:
    """Tests for PurgedKFold cross-validation."""

    def test_basic_split(self, synthetic_features: pd.DataFrame):
        """Test basic K-fold splitting."""
        cv = PurgedKFold(n_splits=5, embargo_pct=0.0)
        splits = cv.split(synthetic_features)

        assert len(splits) == 5, "Should have 5 splits"

        for train_idx, val_idx in splits:
            # No overlap between train and val
            overlap = set(train_idx) & set(val_idx)
            assert len(overlap) == 0, "Train and val should not overlap"

            # All indices should be valid
            assert all(0 <= i < len(synthetic_features) for i in train_idx)
            assert all(0 <= i < len(synthetic_features) for i in val_idx)

    def test_embargo_reduces_train_size(self, synthetic_features: pd.DataFrame):
        """Test that embargo reduces training set size."""
        cv_no_embargo = PurgedKFold(n_splits=5, embargo_pct=0.0)
        cv_with_embargo = PurgedKFold(n_splits=5, embargo_pct=0.05)

        splits_no_embargo = cv_no_embargo.split(synthetic_features)
        splits_with_embargo = cv_with_embargo.split(synthetic_features)

        # With embargo, training sets should be smaller (except possibly first fold)
        for i in range(1, len(splits_no_embargo)):
            train_no_emb = len(splits_no_embargo[i][0])
            train_with_emb = len(splits_with_embargo[i][0])
            assert train_with_emb <= train_no_emb, \
                f"Fold {i}: embargo should reduce train size"

    def test_purging_with_t1(
        self,
        synthetic_features: pd.DataFrame,
        synthetic_labels: pd.Series,
        synthetic_t1: pd.Series,
    ):
        """Test that purging removes overlapping samples."""
        cv = PurgedKFold(n_splits=5, embargo_pct=0.01)

        splits_no_purge = cv.split(synthetic_features, synthetic_labels, t1=None)
        splits_with_purge = cv.split(synthetic_features, synthetic_labels, t1=synthetic_t1)

        # With purging, some folds should have fewer training samples
        # (samples whose t1 extends into validation period are removed)
        total_train_no_purge = sum(len(s[0]) for s in splits_no_purge)
        total_train_with_purge = sum(len(s[0]) for s in splits_with_purge)

        # Purging should reduce total training samples
        assert total_train_with_purge <= total_train_no_purge, \
            "Purging should not increase training samples"

    def test_all_samples_covered_in_validation(self, synthetic_features: pd.DataFrame):
        """Test that all samples appear in validation exactly once."""
        cv = PurgedKFold(n_splits=5, embargo_pct=0.0)
        splits = cv.split(synthetic_features)

        all_val_indices = []
        for _, val_idx in splits:
            all_val_indices.extend(val_idx)

        # Each index should appear exactly once in validation
        assert len(all_val_indices) == len(synthetic_features), \
            "All samples should be in validation exactly once"
        assert len(set(all_val_indices)) == len(synthetic_features), \
            "No duplicate validation samples"

    def test_temporal_order_preserved(self, synthetic_features: pd.DataFrame):
        """Test that validation sets are in temporal order."""
        cv = PurgedKFold(n_splits=5, embargo_pct=0.0)
        splits = cv.split(synthetic_features)

        prev_val_max = -1
        for train_idx, val_idx in splits:
            # Validation indices should be after previous fold's validation
            val_min = min(val_idx)
            assert val_min > prev_val_max, "Validation folds should be temporal"
            prev_val_max = max(val_idx)


# =============================================================================
# TESTS - Metrics Computation
# =============================================================================


class TestComputeMetrics:
    """Tests for compute_metrics function."""

    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([0, 1, -1, 0, 1, -1])
        y_pred = np.array([0, 1, -1, 0, 1, -1])

        metrics = compute_metrics(y_true, y_pred)

        assert metrics.accuracy == 1.0, "Perfect predictions should have 100% accuracy"
        assert metrics.f1_weighted == 1.0, "Perfect predictions should have F1=1"

    def test_random_predictions(self):
        """Test metrics with random predictions."""
        np.random.seed(42)
        y_true = np.random.choice([-1, 0, 1], size=100)
        y_pred = np.random.choice([-1, 0, 1], size=100)

        metrics = compute_metrics(y_true, y_pred)

        # Random predictions should have accuracy around 33%
        assert 0.2 <= metrics.accuracy <= 0.5, \
            f"Random predictions accuracy should be ~33%, got {metrics.accuracy}"

    def test_with_probabilities(self):
        """Test AUC-ROC computation with probabilities."""
        np.random.seed(42)
        y_true = np.array([0, 1, -1, 0, 1, -1, 0, 1, -1, 0])
        y_pred = y_true.copy()  # Perfect predictions

        # Create probability matrix (3 classes)
        n_samples = len(y_true)
        y_proba = np.zeros((n_samples, 3))
        for i, label in enumerate(y_true):
            class_idx = label + 1  # Map -1,0,1 to 0,1,2
            y_proba[i, class_idx] = 0.9
            y_proba[i, :] += 0.033  # Add some noise to other classes
        y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)

        metrics = compute_metrics(y_true, y_pred, y_proba)

        assert metrics.auc_roc_ovr is not None, "AUC-ROC should be computed"
        assert metrics.auc_roc_ovr > 0.8, "Good predictions should have high AUC"

    def test_metrics_to_dict(self):
        """Test that metrics can be converted to dictionary."""
        metrics = EvaluationMetrics(
            accuracy=0.8,
            precision_weighted=0.75,
            recall_weighted=0.78,
            f1_weighted=0.76,
            auc_roc_ovr=0.85,
        )

        result = metrics.to_dict()

        assert isinstance(result, dict)
        assert result["accuracy"] == 0.8
        assert result["auc_roc_ovr"] == 0.85


# =============================================================================
# TESTS - Out-of-Fold Predictions
# =============================================================================


class TestGenerateOOFPredictions:
    """Tests for out-of-fold prediction generation."""

    def test_oof_coverage(
        self,
        synthetic_features: pd.DataFrame,
        synthetic_labels: pd.Series,
        synthetic_t1: pd.Series,
    ):
        """Test that OOF predictions cover most samples."""
        from src.model.ridge_classifier import RidgeClassifierModel

        model_params = {"alpha": 1.0, "random_state": 42}

        oof_preds, oof_proba = generate_oof_predictions(
            X=synthetic_features,
            y=synthetic_labels,
            t1=synthetic_t1,
            model_class=RidgeClassifierModel,
            model_params=model_params,
            n_splits=5,
            embargo_pct=0.01,
        )

        # Check that most samples have predictions
        # Note: With purged K-fold, early folds may be skipped due to insufficient
        # training data, so coverage can be lower than 100%
        valid_preds = ~np.isnan(oof_preds)
        coverage = valid_preds.sum() / len(oof_preds)

        assert coverage > 0.7, f"OOF coverage should be >70%, got {coverage:.1%}"

    def test_oof_no_leakage(
        self,
        synthetic_features: pd.DataFrame,
        synthetic_labels: pd.Series,
        synthetic_t1: pd.Series,
    ):
        """Test that OOF predictions don't leak information."""
        from src.model.ridge_classifier import RidgeClassifierModel

        # Use a simple model
        model_params = {"alpha": 1.0, "random_state": 42}

        oof_preds, _ = generate_oof_predictions(
            X=synthetic_features,
            y=synthetic_labels,
            t1=synthetic_t1,
            model_class=RidgeClassifierModel,
            model_params=model_params,
            n_splits=5,
            embargo_pct=0.01,
        )

        # OOF predictions should not be perfect (that would indicate leakage)
        valid_mask = ~np.isnan(oof_preds)
        y_valid = synthetic_labels.values[valid_mask]
        preds_valid = oof_preds[valid_mask]

        accuracy = (y_valid == preds_valid).mean()

        # If accuracy is too high, there might be leakage
        assert accuracy < 0.95, \
            f"OOF accuracy suspiciously high ({accuracy:.1%}), possible leakage"

    def test_oof_returns_probabilities(
        self,
        synthetic_features: pd.DataFrame,
        synthetic_labels: pd.Series,
        synthetic_t1: pd.Series,
    ):
        """Test that OOF returns probability estimates."""
        from src.model.ridge_classifier import RidgeClassifierModel

        model_params = {"alpha": 1.0, "random_state": 42}

        _, oof_proba = generate_oof_predictions(
            X=synthetic_features,
            y=synthetic_labels,
            t1=synthetic_t1,
            model_class=RidgeClassifierModel,
            model_params=model_params,
            n_splits=5,
            embargo_pct=0.01,
        )

        assert oof_proba is not None, "Should return probabilities"

        # Check probability matrix shape
        n_classes = len(synthetic_labels.unique())
        assert oof_proba.shape[1] == n_classes, \
            f"Proba should have {n_classes} columns"

        # Probabilities should sum to 1 (for valid predictions)
        valid_mask = ~np.isnan(oof_proba[:, 0])
        proba_sums = oof_proba[valid_mask].sum(axis=1)
        np.testing.assert_array_almost_equal(
            proba_sums, np.ones(valid_mask.sum()), decimal=5,
            err_msg="Probabilities should sum to 1"
        )


# =============================================================================
# TESTS - TrainingResult
# =============================================================================


class TestTrainingResult:
    """Tests for TrainingResult dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = TrainingResult(
            model_name="test_model",
            model_params={"alpha": 1.0},
            triple_barrier_params={"pt_mult": 1.0, "sl_mult": 1.0},
            train_samples=1000,
            test_samples=200,
            train_metrics={"accuracy": 0.8},
            test_metrics={"accuracy": 0.75},
            label_distribution_train={"counts": {"-1": 300, "0": 400, "1": 300}},
            label_distribution_test={"counts": {"-1": 60, "0": 80, "1": 60}},
            n_folds=5,
            model_path="/path/to/model.joblib",
            oof_predictions_path="/path/to/oof.parquet",
            test_predictions_path="/path/to/test.parquet",
            labeled_dataset_path="/path/to/labeled_dataset.parquet",
        )

        d = result.to_dict()

        assert d["model_name"] == "test_model"
        assert d["train_samples"] == 1000
        assert d["n_folds"] == 5
        assert d["labeled_dataset_path"] == "/path/to/labeled_dataset.parquet"

    def test_save(self, tmp_path):
        """Test saving results to JSON."""
        result = TrainingResult(
            model_name="test_model",
            model_params={"alpha": 1.0},
            triple_barrier_params={"pt_mult": 1.0, "sl_mult": 1.0},
            train_samples=1000,
            test_samples=200,
            train_metrics={"accuracy": 0.8},
            test_metrics={"accuracy": 0.75},
            label_distribution_train={},
            label_distribution_test={},
            n_folds=5,
            model_path="/path/to/model.joblib",
            oof_predictions_path="/path/to/oof.parquet",
            test_predictions_path="/path/to/test.parquet",
            labeled_dataset_path="/path/to/labeled_dataset.parquet",
        )

        output_path = tmp_path / "results.json"
        result.save(output_path)

        assert output_path.exists(), "Results file should be created"

        # Verify content
        import json
        with open(output_path) as f:
            loaded = json.load(f)

        assert loaded["model_name"] == "test_model"
        assert loaded["train_samples"] == 1000
        assert "labeled_dataset_path" in loaded


# =============================================================================
# INTEGRATION TEST - Mock Full Pipeline
# =============================================================================


class TestFullPipelineMock:
    """Integration tests with mocked data loading."""

    def test_full_pipeline_with_synthetic_data(
        self,
        synthetic_features: pd.DataFrame,
        synthetic_labels: pd.Series,
        synthetic_t1: pd.Series,
        tmp_path,
    ):
        """Test full training pipeline with synthetic data."""
        from src.model.ridge_classifier import RidgeClassifierModel
        from src.labelling.label_primaire.main import (
            generate_oof_predictions,
            compute_metrics,
        )

        # Split into train/test
        n_train = int(len(synthetic_features) * 0.8)

        X_train = synthetic_features.iloc[:n_train]
        X_test = synthetic_features.iloc[n_train:]
        y_train = synthetic_labels.iloc[:n_train]
        y_test = synthetic_labels.iloc[n_train:]
        t1_train = synthetic_t1.iloc[:n_train]

        model_params = {"alpha": 1.0, "random_state": 42}

        # Step 1: Generate OOF predictions
        oof_preds, oof_proba = generate_oof_predictions(
            X=X_train,
            y=y_train,
            t1=t1_train,
            model_class=RidgeClassifierModel,
            model_params=model_params,
            n_splits=5,
            embargo_pct=0.01,
        )

        # Evaluate OOF
        valid_mask = ~np.isnan(oof_preds)
        train_metrics = compute_metrics(
            y_train.values[valid_mask],
            oof_preds[valid_mask].astype(int),
            oof_proba[valid_mask] if oof_proba is not None else None,
        )

        # Step 2: Train final model
        final_model = RidgeClassifierModel(**model_params)
        final_model.fit(X_train, y_train)

        # Step 3: Predict on test
        test_preds = final_model.predict(X_test)
        test_proba = final_model.predict_proba(X_test)

        test_metrics = compute_metrics(y_test.values, test_preds, test_proba)

        # Verify results
        assert train_metrics.accuracy > 0.3, "Train accuracy should be reasonable"
        assert test_metrics.accuracy > 0.3, "Test accuracy should be reasonable"

        # Test should not be much better than train (no leakage)
        assert test_metrics.accuracy <= train_metrics.accuracy + 0.15, \
            "Test accuracy should not be much better than train"

        # Save predictions
        oof_df = pd.DataFrame({
            "datetime": X_train.index,
            "y_true": y_train.values,
            "y_pred": oof_preds,
        })
        oof_path = tmp_path / "oof_predictions.parquet"
        oof_df.to_parquet(oof_path)

        assert oof_path.exists(), "OOF predictions should be saved"
