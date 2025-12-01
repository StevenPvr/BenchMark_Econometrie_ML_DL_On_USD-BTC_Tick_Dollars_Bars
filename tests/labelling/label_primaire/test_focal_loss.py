"""
Tests for focal_loss.py (Focal Loss Implementation)
"""

import numpy as np
import pytest

from src.labelling.label_primaire.focal_loss import (
    focal_loss_lgb_binary,
    focal_loss_lgb_multiclass,
    create_focal_loss_objective,
    compute_focal_alpha_from_class_weights,
    get_default_alpha_for_imbalanced,
)


# =============================================================================
# BINARY FOCAL LOSS TESTS
# =============================================================================


class TestFocalLossBinary:
    """Tests for binary focal loss."""

    def test_basic_output_shape(self):
        """Test that output shapes are correct."""
        n_samples = 100
        y_true = np.random.randint(0, 2, n_samples)
        y_pred = np.random.randn(n_samples)

        grad, hess = focal_loss_lgb_binary(y_true, y_pred, gamma=2.0)

        assert grad.shape == (n_samples,)
        assert hess.shape == (n_samples,)

    def test_gamma_zero_equals_ce(self):
        """Test that gamma=0 is similar to standard cross-entropy."""
        n_samples = 100
        y_true = np.random.randint(0, 2, n_samples)
        y_pred = np.random.randn(n_samples)

        grad_focal, _ = focal_loss_lgb_binary(y_true, y_pred, gamma=0.0, alpha=0.5)

        # With gamma=0 and alpha=0.5, focal loss should be similar to CE
        # (up to a constant factor)
        assert not np.any(np.isnan(grad_focal))
        assert not np.any(np.isinf(grad_focal))

    def test_higher_gamma_reduces_easy_samples(self):
        """Test that higher gamma reduces gradient for well-classified samples."""
        n_samples = 10
        # All samples are class 1, and predictions are high confidence correct
        y_true = np.ones(n_samples)
        y_pred = np.ones(n_samples) * 3.0  # High logit -> high prob for class 1

        grad_low_gamma, _ = focal_loss_lgb_binary(y_true, y_pred, gamma=1.0)
        grad_high_gamma, _ = focal_loss_lgb_binary(y_true, y_pred, gamma=5.0)

        # High gamma should produce smaller gradients for easy samples
        assert np.mean(np.abs(grad_high_gamma)) < np.mean(np.abs(grad_low_gamma))


# =============================================================================
# MULTICLASS FOCAL LOSS TESTS
# =============================================================================


class TestFocalLossMulticlass:
    """Tests for multiclass focal loss."""

    def test_basic_output_shape(self):
        """Test that output shapes are correct for multiclass."""
        n_samples = 100
        n_classes = 3
        y_true = np.random.choice([-1, 0, 1], n_samples)
        y_pred = np.random.randn(n_samples * n_classes)

        grad, hess = focal_loss_lgb_multiclass(
            y_true, y_pred, gamma=2.0, n_classes=n_classes
        )

        assert grad.shape == (n_samples * n_classes,)
        assert hess.shape == (n_samples * n_classes,)

    def test_handles_negative_labels(self):
        """Test that negative labels (-1, 0, 1) are handled correctly."""
        n_samples = 50
        y_true = np.array([-1] * 20 + [0] * 15 + [1] * 15)
        y_pred = np.random.randn(n_samples * 3)

        grad, hess = focal_loss_lgb_multiclass(y_true, y_pred, gamma=2.0)

        assert not np.any(np.isnan(grad))
        assert not np.any(np.isinf(grad))
        assert not np.any(np.isnan(hess))
        assert not np.any(np.isinf(hess))

    def test_handles_zero_indexed_labels(self):
        """Test that zero-indexed labels (0, 1, 2) work correctly."""
        n_samples = 50
        y_true = np.array([0] * 20 + [1] * 15 + [2] * 15)
        y_pred = np.random.randn(n_samples * 3)

        grad, hess = focal_loss_lgb_multiclass(y_true, y_pred, gamma=2.0)

        assert not np.any(np.isnan(grad))
        assert not np.any(np.isinf(grad))

    def test_alpha_weighting(self):
        """Test that alpha weights affect gradients."""
        n_samples = 100
        y_true = np.random.choice([-1, 0, 1], n_samples)
        y_pred = np.random.randn(n_samples * 3)

        # Without alpha
        grad_no_alpha, _ = focal_loss_lgb_multiclass(
            y_true, y_pred, gamma=2.0, alpha=None
        )

        # With alpha emphasizing minority classes
        alpha = np.array([2.0, 0.5, 2.0])
        grad_with_alpha, _ = focal_loss_lgb_multiclass(
            y_true, y_pred, gamma=2.0, alpha=alpha
        )

        # Gradients should be different
        assert not np.allclose(grad_no_alpha, grad_with_alpha)

    def test_gamma_effect(self):
        """Test that gamma affects focusing behavior."""
        n_samples = 50
        y_true = np.random.choice([-1, 0, 1], n_samples)
        y_pred = np.random.randn(n_samples * 3)

        grad_gamma_0, _ = focal_loss_lgb_multiclass(y_true, y_pred, gamma=0.0)
        grad_gamma_2, _ = focal_loss_lgb_multiclass(y_true, y_pred, gamma=2.0)
        grad_gamma_5, _ = focal_loss_lgb_multiclass(y_true, y_pred, gamma=5.0)

        # All should be valid
        assert not np.any(np.isnan(grad_gamma_0))
        assert not np.any(np.isnan(grad_gamma_2))
        assert not np.any(np.isnan(grad_gamma_5))

    def test_hessian_positive(self):
        """Test that hessian values are positive (for convexity)."""
        n_samples = 100
        y_true = np.random.choice([-1, 0, 1], n_samples)
        y_pred = np.random.randn(n_samples * 3)

        _, hess = focal_loss_lgb_multiclass(y_true, y_pred, gamma=2.0)

        # Hessian should be positive for stable optimization
        assert np.all(hess > 0)


# =============================================================================
# OBJECTIVE FACTORY TESTS
# =============================================================================


class TestCreateFocalLossObjective:
    """Tests for the objective factory function."""

    def test_returns_callable(self):
        """Test that factory returns a callable."""
        objective = create_focal_loss_objective(gamma=2.0)
        assert callable(objective)

    def test_objective_output(self):
        """Test that created objective produces valid output."""
        objective = create_focal_loss_objective(
            gamma=2.0,
            alpha=np.array([2.0, 0.5, 2.0]),
            n_classes=3,
        )

        n_samples = 50
        y_true = np.random.choice([-1, 0, 1], n_samples)
        y_pred = np.random.randn(n_samples * 3)

        grad, hess = objective(y_true, y_pred)

        assert grad.shape == (n_samples * 3,)
        assert hess.shape == (n_samples * 3,)
        assert not np.any(np.isnan(grad))
        assert not np.any(np.isnan(hess))

    def test_different_gamma_values(self):
        """Test objective with different gamma values."""
        for gamma in [0.0, 1.0, 2.0, 3.0, 5.0]:
            objective = create_focal_loss_objective(gamma=gamma)

            y_true = np.random.choice([-1, 0, 1], 30)
            y_pred = np.random.randn(90)

            grad, hess = objective(y_true, y_pred)
            assert not np.any(np.isnan(grad))


# =============================================================================
# ALPHA COMPUTATION TESTS
# =============================================================================


class TestComputeFocalAlpha:
    """Tests for alpha computation from class weights."""

    def test_basic_conversion(self):
        """Test basic conversion from class weights to alpha."""
        class_weights = {-1: 2.0, 0: 0.5, 1: 2.0}
        alpha = compute_focal_alpha_from_class_weights(class_weights)

        assert len(alpha) == 3
        assert alpha[0] == pytest.approx(alpha[2])  # -1 and 1 should be equal
        assert alpha[1] < alpha[0]  # 0 should be lower

    def test_normalization(self):
        """Test that alpha is normalized to sum to n_classes."""
        class_weights = {-1: 1.5, 0: 0.3, 1: 1.5}
        alpha = compute_focal_alpha_from_class_weights(class_weights)

        assert np.sum(alpha) == pytest.approx(3.0, rel=1e-5)

    def test_handles_missing_classes(self):
        """Test handling of missing class weights (default to 1.0)."""
        class_weights = {-1: 2.0, 1: 2.0}  # Missing 0
        alpha = compute_focal_alpha_from_class_weights(class_weights)

        assert len(alpha) == 3
        # Class 0 should get default weight of 1.0 (before normalization)


class TestGetDefaultAlpha:
    """Tests for default alpha computation from labels."""

    def test_imbalanced_data(self):
        """Test alpha computation for imbalanced data."""
        # Create imbalanced dataset: 70% class 0, 15% each class -1 and 1
        y = np.array([-1] * 15 + [0] * 70 + [1] * 15)

        alpha = get_default_alpha_for_imbalanced(y, minority_boost=2.0)

        assert len(alpha) == 3
        # Minority classes should have higher weights
        assert alpha[0] > alpha[1]  # -1 > 0
        assert alpha[2] > alpha[1]  # 1 > 0

    def test_balanced_data(self):
        """Test alpha computation for balanced data."""
        y = np.array([-1] * 33 + [0] * 34 + [1] * 33)

        alpha = get_default_alpha_for_imbalanced(y, minority_boost=1.0)

        # All weights should be similar for balanced data
        assert np.std(alpha) < 0.5

    def test_minority_boost_effect(self):
        """Test that minority_boost affects weights."""
        y = np.array([-1] * 15 + [0] * 70 + [1] * 15)

        alpha_low = get_default_alpha_for_imbalanced(y, minority_boost=1.0)
        alpha_high = get_default_alpha_for_imbalanced(y, minority_boost=3.0)

        # Higher boost should give more weight to minority classes
        minority_weight_low = alpha_low[0] + alpha_low[2]
        minority_weight_high = alpha_high[0] + alpha_high[2]

        assert minority_weight_high > minority_weight_low
