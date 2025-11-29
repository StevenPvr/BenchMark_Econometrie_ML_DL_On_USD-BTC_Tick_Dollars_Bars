"""
Tests for main.py (Entry Point)
"""

import pytest
from unittest.mock import MagicMock, patch

# Note: The main.py file actually contains logic that looks like
# "Primary Model Training and Evaluation with Out-of-Fold Predictions"
# but the tests seem to expect `select_mode_interactive`.
# Based on reading the file content, `main.py` seems to focus on "Primary Model Training".
# However, `train.py` also seems to contain training logic.
# Looking at the file I just read: `src/labelling/label_primaire/main.py`
# It has `select_model`, `train_and_evaluate`, `main`.
# It does NOT have `select_mode_interactive` or `run_pipeline`.
# The file content I read for `src/labelling/label_primaire/main.py` actually looks like what I would expect in `train.py`.

# Let's adjust the test to match the ACTUAL content of `src/labelling/label_primaire/main.py`

from src.labelling.label_primaire.main import (
    main,
    select_model,
    train_and_evaluate,
    PurgedKFold,
    generate_oof_predictions,
    get_available_optimized_models,
)

# =============================================================================
# CLI TESTS
# =============================================================================

def test_select_model(mocker):
    # Mock MODEL_REGISTRY
    mocker.patch.dict("src.labelling.label_primaire.main.MODEL_REGISTRY", {"model1": {"dataset": "d1"}})
    mocker.patch("src.labelling.label_primaire.main.get_available_optimized_models", return_value=["model1"])

    # Test valid selection
    mocker.patch("builtins.input", return_value="1")
    assert select_model() == "model1"

def test_main_cli(mocker):
    mocker.patch("src.labelling.label_primaire.main.select_model", return_value="model1")
    # inputs: n_splits, embargo, confirm
    mocker.patch("builtins.input", side_effect=["5", "1.0", "o"])

    mock_train = mocker.patch("src.labelling.label_primaire.main.train_and_evaluate")
    mocker.patch("src.labelling.label_primaire.main.print_results")

    main()
    mock_train.assert_called_once()

# =============================================================================
# PURGED K-FOLD TESTS
# =============================================================================

def test_purged_kfold(mocker):
    import pandas as pd
    import numpy as np

    X = pd.DataFrame(np.random.randn(100, 2), index=pd.date_range("2021-01-01", periods=100))
    t1 = pd.Series(X.index + pd.Timedelta(days=2), index=X.index)

    pkf = PurgedKFold(n_splits=3, embargo_pct=0.01)
    splits = list(pkf.split(X, t1=t1))

    assert len(splits) == 3
    for train, val in splits:
        assert len(train) > 0
        assert len(val) > 0
        # Check orthogonality (standard KFold property)
        assert len(np.intersect1d(train, val)) == 0

# =============================================================================
# WORKFLOW TESTS
# =============================================================================

def test_generate_oof_predictions(mocker):
    import pandas as pd
    import numpy as np

    X = pd.DataFrame(np.random.randn(50, 2))
    y = pd.Series(np.random.randint(0, 2, 50))
    t1 = pd.Series(pd.date_range("2021-01-01", periods=50))

    mock_model_cls = MagicMock()
    mock_instance = MagicMock()
    mock_instance.predict.return_value = np.zeros(10) # size of fold roughly
    mock_instance.predict_proba.return_value = np.zeros((10, 2))
    mock_model_cls.return_value = mock_instance

    # Mock PurgedKFold split to return simple split
    mocker.patch("src.labelling.label_primaire.main.PurgedKFold.split", return_value=[(np.arange(40), np.arange(40, 50))])

    preds, proba = generate_oof_predictions(X, y, t1, mock_model_cls, {}, n_splits=2)

    assert len(preds) == 50
    assert proba.shape == (50, 2)

def test_train_and_evaluate(mocker, tmp_path):
    # Mock everything external
    mocker.patch("src.labelling.label_primaire.main.load_optimized_params", return_value={
        "model_params": {}, "triple_barrier_params": {"pt_mult": 1, "sl_mult": 1, "max_holding": 10}
    })
    mocker.patch("src.labelling.label_primaire.main.load_model_class")

    # Mock data prep
    mock_data = (
        MagicMock(index=[1]), # X_train
        MagicMock(index=[2]), # X_test
        MagicMock(), # y_train
        MagicMock(), # y_test
        MagicMock(), # t1_train
        MagicMock(), # t1_test
    )
    # Give length to mocks
    mock_data[0].__len__ = lambda x: 10
    mock_data[1].__len__ = lambda x: 10
    mock_data[2].__len__ = lambda x: 10
    mock_data[3].__len__ = lambda x: 10
    mock_data[2].value_counts.return_value = MagicMock(to_dict=lambda: {0: 5, 1: 5})
    mock_data[3].value_counts.return_value = MagicMock(to_dict=lambda: {0: 5, 1: 5})

    mocker.patch("src.labelling.label_primaire.main.prepare_data", return_value=mock_data)
    mocker.patch("src.labelling.label_primaire.main.save_labeled_dataset")

    # Mock OOF
    mocker.patch("src.labelling.label_primaire.main.generate_oof_predictions", return_value=(np.zeros(10), np.zeros((10, 2))))

    # Mock metrics
    mock_metrics = MagicMock()
    mock_metrics.to_dict.return_value = {}
    mocker.patch("src.labelling.label_primaire.main.compute_metrics", return_value=mock_metrics)

    result = train_and_evaluate("model1", output_dir=tmp_path)
    assert result.model_name == "model1"
