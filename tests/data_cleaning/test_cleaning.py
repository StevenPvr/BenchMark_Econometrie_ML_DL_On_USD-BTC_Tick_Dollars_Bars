"""Unit tests for src/data_cleaning/cleaning.py."""

from pathlib import Path
from unittest.mock import MagicMock, call

import pandas as pd
import pytest

from src.data_cleaning.cleaning import (
    _drop_duplicates,
    _drop_missing_essentials,
    _filter_price_outliers,
    _load_raw_trades,
    _persist_clean_dataset,
    _strip_unwanted_columns,
    clean_ticks_data,
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

    def test_load_partitioned_dataset(self, mocker):
        """Test loading a partitioned dataset (directory of parquet files)."""
        mock_path = mocker.Mock(spec=Path)
        mock_path.is_dir.return_value = True

        # Mock glob to return sorted parquet files
        p1 = mocker.MagicMock(spec=Path)
        p1.name = "part1.parquet"
        p1.__lt__.return_value = True # p1 < p2

        p2 = mocker.MagicMock(spec=Path)
        p2.name = "part2.parquet"
        p2.__lt__.return_value = False

        mock_path.glob.return_value = [p1, p2]

        # Mock read_parquet
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"a": [3, 4]})

        mock_read_parquet = mocker.patch("pandas.read_parquet", side_effect=[df1, df2])

        mock_shutil_rmtree = mocker.patch("shutil.rmtree")

        # Mock temp file path
        mock_temp_path = mocker.Mock(spec=Path)
        mock_path.with_suffix.return_value = mock_temp_path

        # Mock to_parquet to avoid actual file writing or __fspath__ check
        mocker.patch("pandas.DataFrame.to_parquet")

        # Call function
        result_df = _load_raw_trades(mock_path)

        # Verifications
        assert len(result_df) == 4
        assert result_df["a"].tolist() == [1, 2, 3, 4]

        assert mock_read_parquet.call_count == 2
        mock_read_parquet.assert_has_calls([call(p1), call(p2)])

        # Check saving
        mock_temp_path.rename.assert_called_once_with(mock_path)
        mock_shutil_rmtree.assert_called_once_with(mock_path)

    def test_load_partitioned_dataset_no_files(self, mocker):
        """Test error when partitioned directory is empty."""
        mock_path = mocker.Mock(spec=Path)
        mock_path.is_dir.return_value = True
        mock_path.glob.return_value = []

        with pytest.raises(ValueError, match="No parquet files found"):
            _load_raw_trades(mock_path)

    def test_load_single_file_success(self, mocker):
        """Test loading a single parquet file successfully."""
        mock_path = mocker.Mock(spec=Path)
        mock_path.is_dir.return_value = False

        df = pd.DataFrame({"a": [1, 2]})
        mocker.patch("pandas.read_parquet", return_value=df)

        result = _load_raw_trades(mock_path)
        pd.testing.assert_frame_equal(result, df)

    def test_load_single_file_empty(self, mocker):
        """Test error when loaded dataframe is empty."""
        mock_path = mocker.Mock(spec=Path)
        mock_path.is_dir.return_value = False

        mocker.patch("pandas.read_parquet", return_value=pd.DataFrame())

        with pytest.raises(ValueError, match="Raw trades dataset is empty"):
            _load_raw_trades(mock_path)

    def test_load_single_file_corrupted(self, mocker):
        """Test specific error handling for timeout/corrupted files."""
        mock_path = mocker.Mock(spec=Path)
        mock_path.is_dir.return_value = False
        mock_path.__str__ = lambda x: "dummy_path"

        # Simulate an OSError with "Operation timed out"
        mocker.patch("pandas.read_parquet", side_effect=OSError("Operation timed out"))

        with pytest.raises(RuntimeError, match="Parquet file dummy_path appears to be corrupted"):
            _load_raw_trades(mock_path)

    def test_load_single_file_other_error(self, mocker):
        """Test re-raising of unrelated errors."""
        mock_path = mocker.Mock(spec=Path)
        mock_path.is_dir.return_value = False

        mocker.patch("pandas.read_parquet", side_effect=ValueError("Some other error"))

        with pytest.raises(ValueError, match="Some other error"):
            _load_raw_trades(mock_path)

    def test_load_single_file_oserror_no_timeout(self, mocker):
        """Test re-raising of OSError that is not a timeout."""
        mock_path = mocker.Mock(spec=Path)
        mock_path.is_dir.return_value = False

        mocker.patch("pandas.read_parquet", side_effect=OSError("Disk full"))

        with pytest.raises(OSError, match="Disk full"):
            _load_raw_trades(mock_path)


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

    def test_persist(self, mocker, sample_trades_df):
        """Test saving the dataframe."""
        mock_ensure = mocker.patch("src.data_cleaning.cleaning.ensure_output_dir")
        mock_to_parquet = mocker.patch.object(sample_trades_df, "to_parquet")

        # We also need to check the path constant usage, but it's imported.
        # We can assume the value is correct or check if ensure_output_dir was called with it.
        from src.data_cleaning.cleaning import DATASET_CLEAN_PARQUET

        _persist_clean_dataset(sample_trades_df)

        mock_ensure.assert_called_once_with(DATASET_CLEAN_PARQUET)
        mock_to_parquet.assert_called_once_with(DATASET_CLEAN_PARQUET, index=False)


class TestCleanTicksData:
    """Tests for clean_ticks_data (integration)."""

    def test_clean_ticks_data_flow(self, mocker, sample_trades_df):
        """Test the full flow of clean_ticks_data."""
        # Mock all internal steps
        mocker.patch("src.data_cleaning.cleaning._load_raw_trades", return_value=sample_trades_df)

        # Use side_effect to return modified DFs if needed, or just return the mock object
        # but the function expects DFs.

        # Let's mock the functions to spy on them or ensure they are called
        # But the function modifies the df returned by previous steps.

        # Easiest is to let the pure functions run (they are tested above) and mock load/persist.

        mock_persist = mocker.patch("src.data_cleaning.cleaning._persist_clean_dataset")

        # We need to ensure unwanted columns are removed so persist receives clean df
        # The logic inside clean_ticks_data is:
        # load -> drop missing -> drop dup -> filter outliers -> strip cols -> sort -> persist

        clean_ticks_data()

        # Check persist called
        assert mock_persist.called
        # Get the dataframe passed to persist
        args, _ = mock_persist.call_args
        df_result = args[0]

        assert "id" not in df_result.columns
        assert not df_result.empty

    def test_clean_ticks_data_empty_result(self, mocker):
        """Test error raised if cleaning results in empty dataframe."""
        df = pd.DataFrame({"timestamp": [], "price": [], "amount": []})
        mocker.patch("src.data_cleaning.cleaning._load_raw_trades", return_value=df)

        # Even if load returns empty, the check is at the end.
        # But _load_raw_trades raises if empty. So let's return a non-empty one that gets filtered out completely.

        df_bad = pd.DataFrame({
            "timestamp": [1, 2],
            "price": [None, None], # will be dropped by missing essentials
            "amount": [1, 1]
        })
        mocker.patch("src.data_cleaning.cleaning._load_raw_trades", return_value=df_bad)

        with pytest.raises(ValueError, match="No data remaining after cleaning"):
            clean_ticks_data()
