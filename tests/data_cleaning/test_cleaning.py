"""Unit tests for src/data_cleaning/cleaning.py."""

from unittest import mock

import numpy as np
import pandas as pd
import pytest

from src.data_cleaning.cleaning import (
    _drop_duplicates,
    _drop_missing_essentials,
    _filter_dollar_value_outliers,
    _filter_mad_price_outliers,
    _filter_outliers_robust,
    _filter_price_outliers,
    _filter_rolling_zscore_outliers,
    _filter_volume_outliers,
    _load_raw_trades,
    _persist_clean_dataset,
    _strip_unwanted_columns,
    _validate_numeric_columns,
    clean_ticks_data,
    OutlierReport,
)


@pytest.fixture
def sample_trades_df():
    """Create a sample DataFrame with trades."""
    return pd.DataFrame({
        "timestamp": [1000, 1001, 1002, 1003],
        "id": ["1", "2", "3", "4"],
        "price": [100.0, 101.0, 100.5, 102.0],
        "amount": [1.0, 0.5, 1.2, 0.8],
        "symbol": ["BTC/USD"] * 4,
        "info": ["some info"] * 4
    })


class TestLoadRawTrades:
    """Tests for _load_raw_trades."""

    def test_streams_partitions_and_persists(self, tmp_path):
        """Partitioned parquet files are merged and cached once."""
        partition_dir = tmp_path / "copie_raw"
        partition_dir.mkdir()
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"a": [3, 4]})
        df1.to_parquet(partition_dir / "part-0000.parquet", index=False)
        df2.to_parquet(partition_dir / "part-0001.parquet", index=False)
        output_path = tmp_path / "dataset_raw_consolidated.parquet"

        result_df = _load_raw_trades(partition_dir=partition_dir, output_path=output_path)

        assert output_path.exists()
        assert len(result_df) == 4
        cached_df = pd.read_parquet(output_path)
        pd.testing.assert_frame_equal(result_df, cached_df)

    def test_uses_cached_consolidated_file(self, tmp_path):
        """Existing consolidated parquet is reused when fresher than partitions and use_cache=True."""
        partition_dir = tmp_path / "copie_raw"
        partition_dir.mkdir()
        df = pd.DataFrame({"a": [1, 2]})
        df.to_parquet(partition_dir / "part-0000.parquet", index=False)
        output_path = tmp_path / "dataset_raw_consolidated.parquet"

        _load_raw_trades(partition_dir=partition_dir, output_path=output_path)

        with mock.patch("pyarrow.parquet.ParquetWriter") as writer_mock:
            cached_df = _load_raw_trades(
                partition_dir=partition_dir,
                output_path=output_path,
                use_cache=True,
            )

        writer_mock.assert_not_called()
        pd.testing.assert_frame_equal(cached_df, pd.read_parquet(output_path))

    def test_load_partitioned_dataset_no_files(self, tmp_path):
        """Error when partition directory exists but no partitions are present."""
        empty_dir = tmp_path / "copie_raw"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="No parquet partition files found"):
            _load_raw_trades(partition_dir=empty_dir, output_path=tmp_path / "out.parquet")

    def test_missing_partition_directory(self, tmp_path):
        """Error when partition directory is missing."""
        missing_dir = tmp_path / "copie_raw"
        with pytest.raises(FileNotFoundError, match="Partition directory not found"):
            _load_raw_trades(partition_dir=missing_dir, output_path=tmp_path / "out.parquet")

    def test_replaces_existing_output_directory(self, tmp_path):
        """If output path is a directory, it is removed before writing."""
        partition_dir = tmp_path / "copie_raw"
        partition_dir.mkdir()
        df = pd.DataFrame({"a": [1, 2]})
        df.to_parquet(partition_dir / "part-0000.parquet", index=False)

        output_path = tmp_path / "dataset_raw_consolidated.parquet"
        output_path.mkdir()
        (output_path / "stale.txt").write_text("stale")

        result_df = _load_raw_trades(partition_dir=partition_dir, output_path=output_path)

        assert output_path.is_file()
        assert not (output_path / "stale.txt").exists()
        assert len(result_df) == 2

    def test_corrupted_partitions_raise(self, tmp_path):
        """Timeout-style errors bubble up as RuntimeError for visibility."""
        partition_dir = tmp_path / "copie_raw"
        partition_dir.mkdir()
        df = pd.DataFrame({"a": [1, 2]})
        df.to_parquet(partition_dir / "part-0000.parquet", index=False)
        with mock.patch("pyarrow.dataset.dataset", side_effect=OSError("Operation timed out")):

            with pytest.raises(RuntimeError, match="appear to be corrupted"):
                _load_raw_trades(partition_dir=partition_dir, output_path=tmp_path / "out.parquet")


class TestDropDuplicates:
    """Tests for _drop_duplicates."""

    def test_drop_duplicates_success(self, sample_trades_df):
        """Test that duplicates are removed based on timestamp and id."""
        # Add a duplicate row using iloc with list to preserve dtypes
        dup = sample_trades_df.iloc[[0]]
        df_with_dups = pd.concat([sample_trades_df, dup], ignore_index=True)

        assert len(df_with_dups) == 5
        result = _drop_duplicates(df_with_dups)
        assert len(result) == 4
        pd.testing.assert_frame_equal(result, sample_trades_df)

    def test_drop_duplicates_missing_columns(self, sample_trades_df):
        """Test graceful handling when key columns are missing."""
        df = sample_trades_df.drop(columns=["id"])
        result = _drop_duplicates(df)

        # Let's add a duplicate based on timestamp only
        dup = df.iloc[[0]].copy()
        dup["price"] = 999 # different price, but same timestamp
        df_with_dups = pd.concat([df, dup], ignore_index=True)

        result = _drop_duplicates(df_with_dups)
        # It should drop the second one because subset is just timestamp
        assert len(result) == 4

    def test_no_subset_columns(self):
        """Test when neither timestamp nor id exist."""
        df = pd.DataFrame({"other": [1, 1, 2]})
        result = _drop_duplicates(df)
        assert len(result) == 3


class TestFilterPriceOutliers:
    """Tests for _filter_price_outliers."""

    def test_filter_outliers(self):
        """Test removing ticks with excessive price changes."""
        df = pd.DataFrame({
            "price": [100, 102, 103, 200, 202]
            # 100 -> 102: +2% (ok)
            # 102 -> 103: +1% (ok)
            # 103 -> 200: +94% (bad, > 5%)
            # 200 -> 202: +1% (ok, but previous was removed?
            # Note: pct_change compares to previous row in original DF unless calculated iteratively.
            # Pandas pct_change is vectorized.
            # 103->200 is 0.94. 200->202 is 0.01.
            # So row with 200 should be removed. Row with 202 should be kept (change vs 200 is small)
            # BUT if 200 is bad, maybe 202 is good relative to 200, but relative to 103?
            # The function implements vectorized pct_change.
        })

        # Expected: row with 200 is removed.
        result = _filter_price_outliers(df, max_pct_change=0.1) # 10%

        assert len(result) == 4
        assert 200 not in result["price"].values
        assert 202 in result["price"].values # 202 kept because 202/200 - 1 = 1% < 10%

    def test_missing_price_column(self):
        """Test handling of missing price column."""
        df = pd.DataFrame({"a": [1, 2]})
        result = _filter_price_outliers(df)
        pd.testing.assert_frame_equal(result, df)

    def test_empty_df(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()
        result = _filter_price_outliers(df)
        pd.testing.assert_frame_equal(result, df)


class TestDropMissingEssentials:
    """Tests for _drop_missing_essentials."""

    def test_drop_missing(self, sample_trades_df):
        """Test dropping rows with NaNs in required columns."""
        df = sample_trades_df.copy()
        df.loc[0, "price"] = None
        df.loc[1, "amount"] = None

        result = _drop_missing_essentials(df, required=["price", "amount"])
        assert len(result) == 2
        assert result.iloc[0]["id"] == "3"

    def test_drop_missing_all_good(self, sample_trades_df):
        """Test no rows dropped when data is complete."""
        result = _drop_missing_essentials(sample_trades_df, required=["price"])
        assert len(result) == len(sample_trades_df)


class TestStripUnwantedColumns:
    """Tests for _strip_unwanted_columns."""

    def test_strip_columns(self, sample_trades_df):
        """Test removing specified columns."""
        # sample has id, info, symbol which are unwanted
        result = _strip_unwanted_columns(sample_trades_df)
        assert "id" not in result.columns
        assert "info" not in result.columns
        assert "symbol" not in result.columns
        assert "price" in result.columns
        assert "timestamp" in result.columns


class TestPersistCleanDataset:
    """Tests for _persist_clean_dataset."""

    def test_persist(self, sample_trades_df):
        """Test saving the dataframe."""
        with mock.patch("src.data_cleaning.cleaning.ensure_output_dir") as mock_ensure, \
             mock.patch.object(sample_trades_df, "to_parquet") as mock_to_parquet:
            from src.data_cleaning.cleaning import DATASET_CLEAN_PARQUET

            _persist_clean_dataset(sample_trades_df)

        mock_ensure.assert_called_once_with(DATASET_CLEAN_PARQUET)
        mock_to_parquet.assert_called_once_with(DATASET_CLEAN_PARQUET, index=False)


class TestCleanTicksData:
    """Tests for clean_ticks_data (integration)."""

    def test_clean_ticks_data_flow(self, sample_trades_df, tmp_path):
        """Test the full flow of clean_ticks_data on partitioned input."""
        partition_dir = tmp_path / "copie_raw"
        partition_dir.mkdir()
        sample_trades_df.iloc[:2].to_parquet(partition_dir / "part-0000.parquet", index=False)
        sample_trades_df.iloc[2:].to_parquet(partition_dir / "part-0001.parquet", index=False)
        output_path = tmp_path / "dataset_clean.parquet"

        clean_ticks_data(partition_dir=partition_dir, output_path=output_path)

        assert output_path.exists()
        df_result = pd.read_parquet(output_path)
        assert "id" not in df_result.columns
        assert not df_result.empty

    def test_clean_ticks_data_empty_result(self, tmp_path):
        """Test error raised if cleaning results in empty dataframe."""
        partition_dir = tmp_path / "copie_raw"
        partition_dir.mkdir()
        df_bad = pd.DataFrame({
            "timestamp": [1, 2],
            "price": [np.nan, np.nan],  # NaN values will be dropped by missing essentials
            "amount": [1.0, 1.0]
        })
        df_bad.to_parquet(partition_dir / "part-0000.parquet", index=False)

        with pytest.raises(ValueError, match="No data remaining after cleaning"):
            clean_ticks_data(partition_dir=partition_dir, output_path=tmp_path / "clean.parquet")


# =============================================================================
# TESTS FOR ROBUST OUTLIER DETECTION METHODS (Causal / Expanding)
# =============================================================================


class TestMadPriceOutliers:
    """Tests for _filter_mad_price_outliers (causal expanding version)."""

    def test_removes_extreme_outliers(self):
        """Test that extreme price outliers are removed."""
        # Need variation in prices so MAD is not zero
        np.random.seed(42)
        normal_prices = 100 + np.random.randn(200) * 2  # Prices around 100 with std=2
        outlier = [10000.0]  # Extreme outlier
        more_normal = 100 + np.random.randn(100) * 2
        df = pd.DataFrame({
            "price": np.concatenate([normal_prices, outlier, more_normal]),
        })
        # Use lower threshold to ensure detection
        result, removed = _filter_mad_price_outliers(df, min_periods=50, threshold=3.0)
        assert removed >= 1
        assert 10000.0 not in result["price"].values

    def test_preserves_normal_variation(self):
        """Test that normal price variation is preserved."""
        np.random.seed(42)
        prices = 100 + np.random.randn(500) * 2  # Normal variation
        df = pd.DataFrame({"price": prices})
        result, removed = _filter_mad_price_outliers(df, min_periods=50)
        # Should remove very few if any
        assert removed < 10

    def test_causal_keeps_first_ticks(self):
        """Test that first min_periods ticks are kept (causal behavior)."""
        df = pd.DataFrame({
            "price": [100.0] * 150 + [200.0] * 150,
        })
        result, removed = _filter_mad_price_outliers(df, min_periods=100)
        # First 100 ticks should always be kept (NaN in expanding stats)
        assert len(result) >= 100

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame({"price": pd.Series([], dtype=float)})
        result, removed = _filter_mad_price_outliers(df)
        assert len(result) == 0
        assert removed == 0

    def test_missing_column(self):
        """Test handling of missing price column."""
        df = pd.DataFrame({"other": [1, 2, 3]})
        result, removed = _filter_mad_price_outliers(df)
        assert len(result) == 3
        assert removed == 0


class TestRollingZscoreOutliers:
    """Tests for _filter_rolling_zscore_outliers."""

    def test_removes_local_outliers(self):
        """Test removal of outliers relative to local volatility."""
        np.random.seed(42)
        # Low volatility period, then spike, then low vol again
        prices = np.concatenate([
            100 + np.random.randn(500) * 0.1,  # Low vol
            [150.0],  # Spike
            100 + np.random.randn(500) * 0.1,  # Low vol
        ])
        df = pd.DataFrame({"price": prices})
        result, removed = _filter_rolling_zscore_outliers(df)
        assert removed >= 1

    def test_adapts_to_volatility_regime(self):
        """Test that filter adapts to high volatility periods."""
        np.random.seed(42)
        # High volatility period - larger moves should be tolerated
        prices = 100 + np.random.randn(1000) * 5
        df = pd.DataFrame({"price": np.abs(prices)})  # Ensure positive
        result, removed = _filter_rolling_zscore_outliers(df)
        # Most should be kept despite large moves
        assert len(result) > 900

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame({"price": pd.Series([], dtype=float)})
        result, removed = _filter_rolling_zscore_outliers(df)
        assert len(result) == 0
        assert removed == 0

    def test_insufficient_data(self):
        """Test handling of data smaller than min_periods."""
        df = pd.DataFrame({"price": [100.0] * 50})
        result, removed = _filter_rolling_zscore_outliers(df, min_periods=100)
        # Should return unchanged since not enough data
        assert len(result) == 50
        assert removed == 0


class TestVolumeOutliers:
    """Tests for _filter_volume_outliers."""

    def test_removes_dust_trades(self):
        """Test removal of dust trades (below min volume)."""
        df = pd.DataFrame({
            "amount": [1.0, 0.5, 1e-15, 0.8, 1e-12],
        })
        result, removed_vol, removed_dust = _filter_volume_outliers(df)
        assert removed_dust == 2  # Two dust trades

    def test_removes_extreme_volumes(self):
        """Test removal of extreme volume outliers."""
        # Need variation in volumes so MAD is not zero
        np.random.seed(42)
        normal_volumes = 1.0 + np.random.rand(200) * 0.5  # Volumes around 1.0-1.5
        outlier = [100000.0]  # Extreme outlier
        more_normal = 1.0 + np.random.rand(100) * 0.5
        df = pd.DataFrame({
            "amount": np.concatenate([normal_volumes, outlier, more_normal]),
        })
        result, removed_vol, removed_dust = _filter_volume_outliers(df, min_periods=50, threshold=5.0)
        assert removed_vol >= 1

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame({"amount": pd.Series([], dtype=float)})
        result, removed_vol, removed_dust = _filter_volume_outliers(df)
        assert len(result) == 0
        assert removed_vol == 0
        assert removed_dust == 0

    def test_missing_column(self):
        """Test handling of missing volume column."""
        df = pd.DataFrame({"other": [1, 2, 3]})
        result, removed_vol, removed_dust = _filter_volume_outliers(df)
        assert len(result) == 3
        assert removed_vol == 0
        assert removed_dust == 0


class TestDollarValueOutliers:
    """Tests for _filter_dollar_value_outliers."""

    def test_removes_combined_anomalies(self):
        """Test removal of price*volume anomalies."""
        # Need variation in dollar values so MAD is not zero
        np.random.seed(42)
        normal_prices = 100 + np.random.randn(200) * 2
        normal_amounts = 1.0 + np.random.rand(200) * 0.5
        # Outlier with extreme volume
        outlier_price = [100.0]
        outlier_amount = [100000.0]
        more_normal_prices = 100 + np.random.randn(100) * 2
        more_normal_amounts = 1.0 + np.random.rand(100) * 0.5
        df = pd.DataFrame({
            "price": np.concatenate([normal_prices, outlier_price, more_normal_prices]),
            "amount": np.concatenate([normal_amounts, outlier_amount, more_normal_amounts]),
        })
        # Use lower threshold to ensure detection
        result, removed = _filter_dollar_value_outliers(df, min_periods=50, threshold=5.0)
        assert removed >= 1

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame({
            "price": pd.Series([], dtype=float),
            "amount": pd.Series([], dtype=float),
        })
        result, removed = _filter_dollar_value_outliers(df)
        assert len(result) == 0
        assert removed == 0

    def test_missing_column(self):
        """Test handling of missing columns."""
        df = pd.DataFrame({"price": [100.0, 101.0]})
        result, removed = _filter_dollar_value_outliers(df)
        assert len(result) == 2
        assert removed == 0

    def test_base_usd_pair_uses_amount_notional(self):
        """If USD-like asset is base (e.g., USD/BTC), notional should be volume."""
        df = pd.DataFrame({
            "price": [0.00002, 0.000021, 0.000019, 0.00002, 0.00002],
            "amount": [100.0, 110.0, 105.0, 1_200_000.0, 115.0],
        })
        result, removed = _filter_dollar_value_outliers(
            df, symbol="USD/BTC", min_periods=2, threshold=5.0
        )
        assert removed == 1
        assert 1_200_000.0 not in result["amount"].values


class TestFilterOutliersRobust:
    """Tests for _filter_outliers_robust (integration)."""

    def test_pipeline_skips_expensive_filters(self):
        """Pipeline should skip rolling z-score and dollar value filters."""
        df = pd.DataFrame({
            "price": np.linspace(100, 101, 150),
            "amount": np.ones(150),
        })
        with mock.patch("src.data_cleaning.cleaning._filter_rolling_zscore_outliers") as roll_mock, \
             mock.patch("src.data_cleaning.cleaning._filter_dollar_value_outliers") as dollar_mock:
            _filter_outliers_robust(df)

        roll_mock.assert_not_called()
        dollar_mock.assert_not_called()

    def test_full_pipeline(self):
        """Test complete outlier detection pipeline."""
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame({
            "price": np.abs(100 + np.random.randn(n) * 2),
            "amount": np.abs(1 + np.random.rand(n) * 0.5),
        })
        # Add some outliers
        df.loc[500, "price"] = 500  # Price outlier
        df.loc[600, "amount"] = 1000  # Volume outlier
        df.loc[700, "amount"] = 1e-15  # Dust trade

        result, report = _filter_outliers_robust(df)

        assert report.total_ticks == n
        assert report.final_ticks < n
        assert report.removed_dust_trades >= 1 or report.removed_volume_outliers >= 1
        assert isinstance(report, OutlierReport)

    def test_returns_valid_report(self):
        """Test that OutlierReport is properly populated."""
        df = pd.DataFrame({
            "price": [100.0] * 500,
            "amount": [1.0] * 500,
        })
        _, report = _filter_outliers_robust(df)

        assert report.total_ticks == 500
        assert report.final_ticks <= 500
        # All counts should be non-negative
        assert report.removed_mad_price >= 0
        assert report.removed_rolling_zscore >= 0
        assert report.removed_volume_outliers >= 0
        assert report.removed_dollar_value >= 0
        assert report.removed_dust_trades >= 0

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame({
            "price": pd.Series([], dtype=float),
            "amount": pd.Series([], dtype=float),
        })
        result, report = _filter_outliers_robust(df)
        assert len(result) == 0
        assert report.total_ticks == 0
        assert report.final_ticks == 0

    def test_order_of_operations(self):
        """Test that filters are applied in correct order."""
        # Volume outliers should be removed first
        df = pd.DataFrame({
            "price": [100.0] * 200,
            "amount": [1.0] * 199 + [1e-15],  # One dust trade
        })
        result, report = _filter_outliers_robust(df)
        assert report.removed_dust_trades == 1
        # Dust trade removal happens first, so other stats are cleaner


class TestValidateNumericColumns:
    """Tests for _validate_numeric_columns."""

    def test_valid_numeric_columns(self):
        """Test that valid numeric columns pass validation."""
        df = pd.DataFrame({
            "price": [100.0, 101.0],
            "amount": [1, 2],
        })
        # Should not raise
        _validate_numeric_columns(df, ["price", "amount"])

    def test_missing_column_raises(self):
        """Test that missing column raises KeyError."""
        df = pd.DataFrame({"price": [100.0]})
        with pytest.raises(KeyError, match="Required column 'amount'"):
            _validate_numeric_columns(df, ["price", "amount"])

    def test_non_numeric_column_raises(self):
        """Test that non-numeric column raises TypeError."""
        df = pd.DataFrame({
            "price": ["100", "101"],  # String, not numeric
        })
        with pytest.raises(TypeError, match="must be numeric"):
            _validate_numeric_columns(df, ["price"])

    def test_integer_column_passes(self):
        """Test that integer columns are considered numeric."""
        df = pd.DataFrame({
            "count": [1, 2, 3],
        })
        # Should not raise
        _validate_numeric_columns(df, ["count"])


class TestOutlierReport:
    """Tests for OutlierReport dataclass."""

    def test_log_summary_no_error(self):
        """Test that log_summary runs without errors."""
        report = OutlierReport(
            total_ticks=1000,
            removed_mad_price=10,
            removed_rolling_zscore=5,
            removed_volume_outliers=3,
            removed_dollar_value=2,
            removed_dust_trades=8,
            final_ticks=972,
        )
        # Should not raise
        report.log_summary()

    def test_log_summary_zero_ticks(self):
        """Test that log_summary handles zero ticks gracefully."""
        report = OutlierReport(total_ticks=0, final_ticks=0)
        # Should not raise (division by zero protection)
        report.log_summary()
