"""Tests for src.labelling.label_meta.utils."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.labelling.label_meta.utils import (
    MODEL_REGISTRY,
    MetaOptimizationConfig,
    MetaOptimizationResult,
    load_model_class,
    get_dataset_for_model,
    load_dollar_bars,
    load_primary_model,
    get_available_primary_models,
    get_daily_volatility,
    set_vertical_barriers_meta,
    compute_side_adjusted_barriers,
    get_barrier_touches_for_side,
    find_first_touch_time,
    compute_meta_label,
)

class TestMetaOptimizationClasses:
    def test_config_defaults(self):
        """Test config defaults."""
        config = MetaOptimizationConfig("primary", "meta")
        assert config.n_trials == 50
        assert config.random_state == 42 # Assuming DEFAULT_RANDOM_STATE is 42

    def test_result_serialization(self, tmp_path):
        """Test result serialization."""
        result = MetaOptimizationResult(
            "primary", "meta", {"p": 1}, {"tp": 2}, 0.8, "accuracy", 10
        )
        path = tmp_path / "result.json"
        result.save(path)

        assert path.exists()
        import json
        with open(path) as f:
            data = json.load(f)
            assert data["primary_model_name"] == "primary"

class TestDataLoading:
    def test_load_model_class(self):
        """Test loading model class."""
        # Test with a known model in registry
        # We can pick 'lightgbm' if available, or mock registry
        # The registry points to src.model...
        # Let's assume the registry is correct and test if it tries to import

        # We can try to load a model that we know exists or mock the import
        # But import logic is hard to mock easily.
        # Let's trust it works if we can load one.

        # Or better: mock MODEL_REGISTRY for this test
        # We need to patch it in the module
        pass
        # Since patching global dict in module is tricky if it's already imported
        # But we can try loading 'lightgbm' and assert it returns a class
        # assuming lightgbm is installed (it is) and code exists.

        # Test invalid model
        with pytest.raises(ValueError, match="Unknown model"):
            load_model_class("invalid_model")

    def test_get_dataset_for_model(self, mocker, tmp_path):
        """Test dataset loading."""
        # Create dummy parquet file
        dummy_df = pd.DataFrame({"col": [1, 2, 3]})
        parquet_path = tmp_path / "test.parquet"
        dummy_df.to_parquet(parquet_path)

        mocker.patch("src.labelling.label_meta.utils.DATASET_FEATURES_FINAL_PARQUET", parquet_path)
        mocker.patch("src.labelling.label_meta.utils.DATASET_FEATURES_LINEAR_FINAL_PARQUET", parquet_path)
        mocker.patch("src.labelling.label_meta.utils.DATASET_FEATURES_LSTM_FINAL_PARQUET", parquet_path)

        df = get_dataset_for_model("lightgbm")  # tree dataset
        assert isinstance(df, pd.DataFrame)

        with pytest.raises(ValueError):
            get_dataset_for_model("invalid")

    def test_load_dollar_bars(self, mocker):
        """Test dollar bars loading."""
        mocker.patch("src.labelling.label_meta.utils.DOLLAR_BARS_PARQUET", Path("dummy.parquet"))
        mocker.patch("pathlib.Path.exists", return_value=True)

        df = pd.DataFrame({
            "datetime_close": pd.date_range("2024-01-01", periods=10),
            "close": np.random.randn(10) + 100
        })
        mocker.patch("pandas.read_parquet", return_value=df)

        bars = load_dollar_bars()
        assert "log_return" in bars.columns

class TestVolatility:
    def test_get_daily_volatility(self):
        """Test volatility calculation."""
        close = pd.Series(np.cumprod(1 + np.random.normal(0, 0.01, 100)),
                          index=pd.date_range("2024-01-01", periods=100))
        vol = get_daily_volatility(close, span=10)
        assert len(vol) == 100
        assert not vol.isnull().all()

class TestBarriers:
    def test_set_vertical_barriers_meta(self):
        """Test vertical barriers."""
        dates = pd.date_range("2024-01-01", periods=10)
        close_idx = dates
        events = pd.DataFrame(index=dates[:5])

        events = set_vertical_barriers_meta(events, close_idx, max_holding=2)

        # For first event (index 0), t1 should be index 2
        assert events.iloc[0]["t1"] == dates[2]

    def test_compute_side_adjusted_barriers(self):
        """Test barrier adjustment."""
        events = pd.DataFrame({
            "trgt": [0.01, 0.01],
            "side": [1, -1] # Long, Short
        })

        events = compute_side_adjusted_barriers(events, pt_mult=2.0, sl_mult=1.0)

        # Long: pt = 2 * 0.01 * 1 = 0.02. sl = -1 * 0.01 * 1 = -0.01
        assert events.iloc[0]["pt"] == 0.02
        assert events.iloc[0]["sl"] == -0.01

        # Short: pt = 2 * 0.01 * -1 = -0.02. sl = -1 * 0.01 * -1 = 0.01
        assert events.iloc[1]["pt"] == -0.02
        assert events.iloc[1]["sl"] == 0.01

    def test_get_barrier_touches_for_side_long(self):
        """Test barrier touches for long side."""
        path = pd.Series([0.0, 0.01, 0.03, -0.02], index=range(4))
        # Long
        pt = 0.02
        sl = -0.01

        pt_touches, sl_touches = get_barrier_touches_for_side(path, 1, pt, sl)

        assert 2 in pt_touches.index # 0.03 >= 0.02
        assert 3 in sl_touches.index # -0.02 <= -0.01

    def test_find_first_touch_time(self):
        """Test finding first touch."""
        pt_touches = pd.Series([1], index=[pd.Timestamp("2024-01-02")])
        sl_touches = pd.Series([1], index=[pd.Timestamp("2024-01-01")])

        # SL happened first
        assert find_first_touch_time(pt_touches, sl_touches) == pd.Timestamp("2024-01-01")

    def test_compute_meta_label(self):
        """Test meta label computation."""
        # Profitable long
        assert compute_meta_label(0.05, 1) == 1
        # Losing long
        assert compute_meta_label(-0.05, 1) == 0
        # Profitable short
        assert compute_meta_label(-0.05, -1) == 1
        # Losing short
        assert compute_meta_label(0.05, -1) == 0

    def test_get_available_primary_models(self, tmp_path, mocker):
        """Test listing primary models."""
        mocker.patch("src.labelling.label_meta.utils.LABEL_PRIMAIRE_TRAIN_DIR", tmp_path)

        # Create dummy structure
        (tmp_path / "model1").mkdir()
        (tmp_path / "model1" / "model1_model.joblib").touch()
        (tmp_path / "model2").mkdir() # No joblib
        (tmp_path / "file.txt").touch() # Not a dir

        models = get_available_primary_models()
        assert "model1" in models
        assert "model2" not in models
        assert "file.txt" not in models

    def test_load_primary_model(self, tmp_path, mocker):
        """Test loading primary model."""
        mocker.patch("src.labelling.label_meta.utils.LABEL_PRIMAIRE_TRAIN_DIR", tmp_path)

        # Mock BaseModel.load
        mock_load = mocker.patch("src.model.base.BaseModel.load")

        # Test success
        (tmp_path / "model1").mkdir()
        (tmp_path / "model1" / "model1_model.joblib").touch()

        load_primary_model("model1")
        mock_load.assert_called_once()

        # Test failure
        with pytest.raises(FileNotFoundError):
            load_primary_model("nonexistent")

    def test_set_vertical_barriers_meta_missing_index(self):
        """Test vertical barriers with missing index in close prices."""
        dates = pd.date_range("2024-01-01", periods=10)
        close_idx = dates[:5] # Shorter than events
        events = pd.DataFrame(index=dates) # Some dates not in close_idx

        events = set_vertical_barriers_meta(events, close_idx, max_holding=2)

        # Events not in close_idx should have NaT t1
        assert pd.isna(events.loc[dates[6], "t1"])
