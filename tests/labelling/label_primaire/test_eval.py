"""
Tests for eval.py (Evaluation Module)
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock
import matplotlib.pyplot as plt

from src.labelling.label_primaire.eval import (
    evaluate_model,
    compute_classification_metrics,
    ClassificationMetrics,
    EvaluationResult,
    extract_probabilities,
)

# =============================================================================
# DATA FIXTURES
# =============================================================================

@pytest.fixture
def evaluation_data():
    np.random.seed(42)
    n_samples = 100
    y_true = np.random.randint(0, 2, n_samples)
    y_pred = np.random.randint(0, 2, n_samples)
    y_prob = np.random.rand(n_samples, 2) # Probabilities for class 0 and 1

    # Normalize probabilities
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)

    return y_true, y_pred, y_prob

# =============================================================================
# EVALUATION METRICS TESTS
# =============================================================================

def test_compute_classification_metrics(evaluation_data):
    y_true, y_pred, y_prob = evaluation_data

    metrics = compute_classification_metrics(y_true, y_pred, y_prob)

    assert isinstance(metrics, ClassificationMetrics)
    assert 0 <= metrics.accuracy <= 1
    assert 0 <= metrics.f1_macro <= 1
    assert -1 <= metrics.mcc <= 1
    assert metrics.auc_roc_macro is not None
    assert metrics.log_loss_value is not None

    d = metrics.to_dict()
    assert isinstance(d, dict)
    assert d["accuracy"] == metrics.accuracy

    summary = metrics.summary()
    assert "accuracy" in summary

def test_compute_classification_metrics_no_proba(evaluation_data):
    y_true, y_pred, _ = evaluation_data

    metrics = compute_classification_metrics(y_true, y_pred, None)
    assert metrics.auc_roc_macro is None
    assert metrics.log_loss_value is None

# =============================================================================
# DATA HELPERS
# =============================================================================

def test_extract_probabilities():
    df = pd.DataFrame({
        "pred": [1, 0],
        "proba_class_0": [0.1, 0.9],
        "proba_class_1": [0.9, 0.1]
    })

    probs = extract_probabilities(df)
    assert probs is not None
    assert probs.shape == (2, 2)
    assert probs[0, 0] == 0.1

    # No probs
    df_none = pd.DataFrame({"pred": [1, 0]})
    assert extract_probabilities(df_none) is None

# =============================================================================
# FULL EVALUATION WORKFLOW
# =============================================================================

def test_evaluate_model(mocker, evaluation_data, tmp_path):
    y_true, y_pred, y_prob = evaluation_data

    # Mock file loading
    mock_oof = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "proba_class_0": y_prob[:, 0],
        "proba_class_1": y_prob[:, 1]
    })
    mock_test = mock_oof.copy()

    mocker.patch(
        "src.labelling.label_primaire.eval.load_predictions",
        return_value=(mock_oof, mock_test)
    )

    # Mock paths
    mocker.patch(
        "src.labelling.label_primaire.eval.LABEL_PRIMAIRE_TRAIN_DIR",
        tmp_path
    )

    # Create evaluation result
    result = evaluate_model("lightgbm")

    assert isinstance(result, EvaluationResult)
    assert result.model_name == "lightgbm"
    assert result.train_samples == len(y_true)

    # Test summary
    summary = result.comparison_summary()
    assert "train" in summary
    assert "test" in summary

# =============================================================================
# EVALUATION RESULT CLASS
# =============================================================================

def test_evaluation_result_serialization(tmp_path):
    res = EvaluationResult(
        model_name="test",
        train_metrics={"acc": 0.8},
        test_metrics={"acc": 0.7},
        train_samples=100,
        test_samples=50,
        label_distribution_train={},
        label_distribution_test={},
        model_path="model.joblib",
        oof_predictions_path="oof.parquet",
        test_predictions_path="test.parquet"
    )

    d = res.to_dict()
    assert d["model_name"] == "test"

    p = tmp_path / "result.json"
    res.save(p)
    assert p.exists()
